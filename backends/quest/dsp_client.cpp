// SPDX-License-Identifier: GPL-3.0-or-later
// A bounded client plugin for the existing MIX private-PD hexagonrpcd.
// Meta wire layouts are recovered from libhexagon.so, SHA-256 23d722d3...a405.
// Two persistent DSP graphs; no CPU inference or camera access.
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <misc/fastrpc.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#include <array>
#include <memory>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>

// nnGraph.setInputs and setOutputs require 128-byte aligned buffers with
// round-up plus 128-byte tail space. Keep this allocator for every RPC buffer.
template<class T> struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() = default;
    template<class U> AlignedAllocator(const AlignedAllocator<U> &) {}
    T *allocate(size_t n) {
        if (n > std::numeric_limits<size_t>::max()/sizeof(T)) throw std::bad_alloc();
        void *p = nullptr;
        if (posix_memalign(&p, 128, std::max(size_t(128), n*sizeof(T)))) throw std::bad_alloc();
        return static_cast<T *>(p);
    }
    void deallocate(T *p, size_t) { std::free(p); }
};
template<class T, class U> bool operator==(const AlignedAllocator<T> &, const AlignedAllocator<U> &) { return true; }
template<class T, class U> bool operator!=(const AlignedAllocator<T> &, const AlignedAllocator<U> &) { return false; }
using Bytes = std::vector<uint8_t, AlignedAllocator<uint8_t>>;
using Buffers = std::vector<Bytes>;
static double now_ms() {
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts)) throw std::runtime_error("clock_gettime");
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static void word(Bytes &b, uint32_t w) {
    for (unsigned j = 0; j < 4; ++j) b.push_back(static_cast<uint8_t>(w >> (8*j)));
}
static Bytes words(std::initializer_list<uint32_t> v) {
    Bytes b; for (auto w : v) word(b, w); return b;
}
static uint32_t at(const Bytes &b, size_t off) {
    if (off > b.size() || b.size() - off < 4) throw std::runtime_error("short wire word");
    uint32_t w = 0; for (unsigned j = 0; j < 4; ++j) w |= uint32_t(b[off+j]) << (8*j);
    return w;
}
static Bytes slice(const Bytes &b, size_t off, size_t len) {
    if (off > b.size() || len > b.size() - off) throw std::runtime_error("short packet body");
    return Bytes(b.begin()+off, b.begin()+off+len);
}
static Bytes load(const std::string &p, size_t limit) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("open input: " + p);
    auto n = f.tellg();
    if (n < 0 || uint64_t(n) > limit) throw std::runtime_error("input size: " + p);
    Bytes b(static_cast<size_t>(n)); f.seekg(0);
    if (!f.read(reinterpret_cast<char *>(b.data()), n)) throw std::runtime_error("read: " + p);
    return b;
}
static int invoke(int fd, uint32_t handle, unsigned method, Buffers &in, Buffers &out) {
    if (method > 31 || in.size() > 255 || out.size() > 255) throw std::runtime_error("scalar bounds");
    std::vector<fastrpc_invoke_args> a;
    for (auto *buffers : {&in, &out}) for (auto &b : *buffers) {
        fastrpc_invoke_args arg{};
        arg.ptr = b.empty() ? 0 : reinterpret_cast<uintptr_t>(b.data());
        arg.length = b.size(); arg.fd = -1; a.push_back(arg);
    }
    fastrpc_invoke req{};
    req.handle = handle;
    req.sc = (method << 24) | (uint32_t(in.size()) << 16) | (uint32_t(out.size()) << 8);
    req.args = reinterpret_cast<uintptr_t>(a.data());
    errno = 0;
    int rc = ioctl(fd, FASTRPC_IOCTL_INVOKE, &req);
    if (rc) std::fprintf(stderr, "stage=rpc_error handle=0x%x scalar=0x%x rc=%d errno=%d\n", handle, req.sc, rc, rc == -1 ? errno : 0);
    return rc;
}
static void require(int rc, const char *stage) {
    std::printf("stage=%s rc=%d\n", stage, rc);
    if (rc) throw std::runtime_error(stage);
}
static int policy_env(const char *name,int fallback,int minimum,int maximum) {
    const char *value=std::getenv(name);if(!value)return fallback;
    char *end=nullptr;errno=0;long n=std::strtol(value,&end,10);
    if(errno||*end||n<minimum||n>maximum)throw std::runtime_error(std::string("invalid policy ")+name);
    return int(n);
}
struct RunPolicy {
    int threads=policy_env("QUEST_PIPELINE_THREADS",-1,-1,4);
    int priority=policy_env("QUEST_PIPELINE_PRIORITY",0,0,2);
    int vtcm_bytes=policy_env("QUEST_PIPELINE_VTCM_BYTES",-1,-1,262144);
    RunPolicy(){if(!threads)throw std::runtime_error("threads must be-1 or1..4");}
};
struct Module {
    RunPolicy policy;
    int fd; uint32_t handle = 0, graph = 0; bool opened = false, initialized = false, started = false;
    std::vector<uint32_t> graphs;
    explicit Module(int value):fd(value) {}
    void dump_log() {
#ifdef XR_QUEST_DIAGNOSTIC_INNER_LOG
        // Only with the hash-verified diagnostic COPY of the DSP library.
        // RPC44 encodes the inner graph ID in capacity; the production library
        // has a different GetLog contract. Execute has returned synchronously,
        // so no graph execution or destruction races these draining reads.
        unsigned found=0;
        for (uint32_t id=1;id<=128;++id) {
            unsigned capacity=id*128;
            Buffers in{words({44,capacity})},out{Bytes(capacity)};
            int rc=invoke(fd,handle,31,in,out);
            if (!rc) {
                ++found;
                std::printf("stage=legacy_log gid=%u rc=0 capacity=%u text=%.*s\n",id,capacity,int(capacity-1),out[0].data());
            }
        }
        std::printf("stage=legacy_logs_complete valid_graphs=%u scope=owned_private_PD_diagnostic_library_only\n",found);
#else
        Buffers in{words({44, 16384})}, out{Bytes(16384)};
        int rc = invoke(fd, handle, 31, in, out);
        std::printf("stage=nn_log rc=%d text=%.*s\n", rc, 16383, out[0].data());
#endif
    }
    ~Module() {
        for (auto it=graphs.rbegin();it!=graphs.rend();++it) {
            Buffers in{words({46, *it})}, out;
            int rc = invoke(fd, handle, 31, in, out);
            std::printf("stage=graph_destroy graph=%u rc=%d\n", *it, rc);
        }
        if (started) {
            Buffers in{words({41})}, out;
            std::printf("stage=nn_stop rc=%d\n", invoke(fd, handle, 31, in, out));
        }
        if (opened) {
            // Same wire shape as canonical libhexagonrpc remotectl_close.
            Buffers in{words({handle, 512})}, out{Bytes(4), Bytes(512)};
            int rc = invoke(fd, 0, 1, in, out);
            std::printf("stage=module_close rc=%d loader_error=%u\n", rc, at(out[0], 0));
        }
    }
    void open() {
        const std::string uri = "file:///libhexagon_skel.so?hexagon_skel_handle_invoke&_modver=1.0";
        Bytes name(uri.begin(), uri.end()); name.push_back(0);
        Buffers in{words({uint32_t(name.size()), 1024}), name}, out{Bytes(8), Bytes(1024)};
        int rc = invoke(fd, 0, 0, in, out);
        std::printf("stage=module_open rc=%d loader_error=%u detail=%.*s\n", rc, at(out[0], 4), 1023, out[1].data());
        if (rc || at(out[0], 4)) throw std::runtime_error("Quest DSP module was not loaded");
        handle = at(out[0], 0); opened = true;
        for (unsigned method : {3u, 4u}) {
            Buffers vi, vo{Bytes(4)};
            require(invoke(fd, handle, method, vi, vo), method == 3 ? "meta_abi" : "dsp_arch");
            std::printf("stage=version method=%u value=0x%x\n", method, at(vo[0], 0));
        }
        Buffers ri{words({256})}, ro{Bytes(256)};
        require(invoke(fd, handle, 2, ri, ro), "dsp_revision");
        std::printf("stage=dsp_revision_string value=%.*s\n", 255, ro[0].data());
    }
    void init() {
        Buffers in, out;
        if (!started) {
            in={words({40})};
            require(invoke(fd, handle, 31, in, out), "nn_start");
            started = true;
        }
        unsigned debug=1;
        if(const char *value=std::getenv("QUEST_PIPELINE_DEBUG_LEVEL")) {
            char *end=nullptr;unsigned long parsed=std::strtoul(value,&end,10);
            if(*end||parsed<1||parsed>9)throw std::runtime_error("debug level must be1..9");
            debug=unsigned(parsed);
        }
        std::printf("stage=debug_level meta=%u internal_nnlib=%u\n",debug,debug>2?debug-2:0);
        in = {words({42, debug})};
        require(invoke(fd, handle, 31, in, out), "nn_debug_level");
        in = {words({45})}; out = {Bytes(4)};
        require(invoke(fd, handle, 31, in, out), "nn_init");
        graph = at(out[0], 0); initialized = true; graphs.push_back(graph);
        std::printf("stage=graph_id value=%u\n", graph);
    }
    void build(const Bytes &p) {
        if (p.size() < 8 || std::memcmp(p.data(), "QHF1", 4)) throw std::runtime_error("packet magic");
        size_t pos = 8;
        uint32_t records = at(p, 4);
        if (!records || records > 2048) throw std::runtime_error("record count");
        for (uint32_t j = 0; j < records; ++j) {
            uint32_t kind = at(p, pos), id = at(p, pos+4), len = at(p, pos+8); pos += 12;
            Bytes b = slice(p, pos, len); pos += len;
            Buffers in, out;
            if (kind == 0) {
                auto count = at(b, 40);
                if (b.size() != 44u + count) throw std::runtime_error("constant size");
                Bytes prim = words({52, graph, id});
                auto desc = slice(b, 0, 40); prim.insert(prim.end(), desc.begin(), desc.end()); word(prim, count);
                in = {prim, slice(b, 44, count)};
            } else if (kind == 1) {
                auto n = at(b, 0), pad = at(b, 4), ni = at(b, 8), no = at(b, 12);
                if (!n || n > 128 || ni > 16 || no > 4 || b.size() != 16u+n+ni*8u+no*40u || b[15+n]) throw std::runtime_error("node size");
                in = {words({50, graph, id, n, pad, ni, no}), slice(b, 16, n),
                      slice(b, 16+n, ni*8), slice(b, 16+n+ni*8, no*40)};
            } else throw std::runtime_error("record kind");
            int rc = invoke(fd, handle, 31, in, out);
            std::printf("stage=append kind=%u node=%u rc=%d\n", kind, id, rc);
            if (rc) { dump_log(); throw std::runtime_error("append rejected"); }
        }
        if (pos != p.size()) throw std::runtime_error("packet trailing bytes");
        // Both Prepare and Execute reapply the thread hint. Keeping Execute
        // at -1 would reset a Prepare-only limit to four workers.
        std::printf("stage=run_policy graph=%u prepare_threads=%d execute_threads=%d priority=%d vtcm_hint_bytes=%d vtcm_zero_is_not_disabled=1\n",graph,policy.threads,policy.threads,policy.priority,policy.vtcm_bytes);
        Buffers in{words({55, graph, uint32_t(policy.priority), uint32_t(policy.threads), uint32_t(policy.vtcm_bytes), 0})}, out{Bytes(8)};
        double t = now_ms(); int rc = invoke(fd, handle, 31, in, out); double elapsed = now_ms()-t;
        std::printf("stage=prepare rc=%d host_rpc_ms=%.6f raw_result0=%u raw_result1=%u\n", rc, elapsed, at(out[0], 0), at(out[0], 4));
        require(rc, "prepare_result");
        if (at(out[0], 0)) { dump_log(); throw std::runtime_error("DSP graph prepare error, see raw_result0/node in log"); }
    }
};

static uint64_t monotonic_ns() {
    timespec t{}; if(clock_gettime(CLOCK_MONOTONIC,&t)) throw std::runtime_error("monotonic clock");
    return uint64_t(t.tv_sec)*1000000000+uint64_t(t.tv_nsec);
}
static void word64(Bytes &b,uint64_t x) {word(b,uint32_t(x));word(b,uint32_t(x>>32));}
static uint64_t at64(const Bytes &b,size_t offset) {return uint64_t(at(b,offset))|(uint64_t(at(b,offset+4))<<32);}
static uint32_t padded(uint32_t n) {return ((n+127)/128)*128+128;}
struct Execution {
    uint64_t start_ns=0,end_ns=0;
    double rpc_ms=0;
    float min=0,max=0;
    Bytes data;
};
struct Graph {
    Module &m; uint32_t gid; std::array<uint32_t,4> output_shape;
    uint32_t input_bytes,output_bytes;
    Buffers in,out;
    Graph(Module &module, std::array<uint32_t,4> input, std::array<uint32_t,4> output):m(module),gid(module.graph),output_shape(output) {
        input_bytes=1;output_bytes=1;
        for(auto n:input) input_bytes*=n;
        for(auto n:output) output_bytes*=n;
        Bytes meta=words({input[0],input[1],input[2],input[3],0,0,0,0,4,1,padded(input_bytes)});
        in={words({56,gid,1,3,uint32_t(module.policy.priority),uint32_t(module.policy.threads),0,0}),meta,Bytes(padded(input_bytes)),words({padded(output_bytes),256,256})};
        out={Bytes(8),Bytes(120),Bytes(padded(output_bytes)),Bytes(256),Bytes(256)};
    }
    Execution execute(const Bytes &data) {
        if(data.size()!=input_bytes) throw std::runtime_error("input payload length");
        std::copy(data.begin(),data.end(),in[2].begin());
        for(auto &b:out) std::fill(b.begin(),b.end(),0xa5);
        Execution e;e.start_ns=monotonic_ns();double raw=now_ms();
        int rc=invoke(m.fd,m.handle,31,in,out);
        e.rpc_ms=now_ms()-raw;e.end_ns=monotonic_ns();
        if(rc||at(out[0],0)) {
            std::fprintf(stderr,"stage=graph_execute_error graph=%u transport=%d code=%u node=%u\n",gid,rc,at(out[0],0),at(out[0],4));
            m.dump_log();throw std::runtime_error("graph execution failed");
        }
        for(size_t j=0;j<4;++j) if(at(out[1],j*4)!=output_shape[j]) throw std::runtime_error("output shape");
        // nnOpOutputExecute returns dims/rank, leaves dtype at +36 undefined.
        if(at(out[1],32)!=4) throw std::runtime_error("output rank mismatch");
        for(unsigned j=1;j<3;++j) if(at(out[1],40*j)!=1||at(out[1],40*j+32)!=1) throw std::runtime_error("output range shape");
        std::memcpy(&e.min,out[3].data(),4);std::memcpy(&e.max,out[4].data(),4);
        if(!std::isfinite(e.min)||!std::isfinite(e.max)||e.max<=e.min) throw std::runtime_error("output range");
        e.data=slice(out[2],0,output_bytes);return e;
    }
};
// Three independent float32 outputs, preserving original outer DETV62 graph.
struct DetectorGraph {
    Module &m;uint32_t gid;
    static constexpr uint32_t input_bytes=4*480*640;
    Buffers in,out;
    explicit DetectorGraph(Module &module):m(module),gid(module.graph) {
        Bytes meta=words({4,1,480,640,0,0,0,0,4,1,padded(input_bytes)});
        in={words({56,gid,1,3,uint32_t(module.policy.priority),uint32_t(module.policy.threads),0,0}),meta,Bytes(padded(input_bytes)),words({256,256,256})};
        out={Bytes(8),Bytes(120),Bytes(256),Bytes(256),Bytes(256)};
    }
    Execution execute(const Bytes &data) {
        if(data.size()!=input_bytes)throw std::runtime_error("detector input payload length");
        std::copy(data.begin(),data.end(),in[2].begin());
        for(auto &b:out)std::fill(b.begin(),b.end(),0xa5);
        Execution e;e.start_ns=monotonic_ns();double raw=now_ms();
        int rc=invoke(m.fd,m.handle,31,in,out);
        e.rpc_ms=now_ms()-raw;e.end_ns=monotonic_ns();
        if(rc||at(out[0],0)) {
            std::fprintf(stderr,"stage=detector_execute_error graph=%u transport=%d code=%u node=%u\n",gid,rc,at(out[0],0),at(out[0],4));
            m.dump_log();throw std::runtime_error("detector graph execution failed");
        }
        const uint32_t ranks[]={3,2,2},sizes[]={64,32,32};
        for(unsigned j=0;j<3;++j) {
            if(at(out[1],j*40+32)!=ranks[j])throw std::runtime_error("detector output rank mismatch");
            for(unsigned k=0;k<ranks[j];++k)if(at(out[1],j*40+k*4)!=(k?2u:4u))throw std::runtime_error("detector output shape");
            for(unsigned k=0;k<sizes[j];k+=4) {
                float value;std::memcpy(&value,out[j+2].data()+k,4);
                if(!std::isfinite(value))throw std::runtime_error("nonfinite detector output");
            }
            e.data.insert(e.data.end(),out[j+2].begin(),out[j+2].begin()+sizes[j]);
        }
        return e;
    }
};
struct Descriptor {
    int fd=-1;
    explicit Descriptor(int value):fd(value) {}
    ~Descriptor(){if(fd>=0) close(fd);}
    Descriptor(const Descriptor&)=delete;
};
struct SocketPath {
    std::string path;
    ~SocketPath(){unlink(path.c_str());}
};
static bool readable(int fd,unsigned timeout) {
    pollfd p{fd,POLLIN,0};int rc;
    do {rc=poll(&p,1,int(timeout));}while(rc<0&&errno==EINTR);
    if(rc<0)throw std::runtime_error("socket poll");
    return rc>0&&(p.revents&(POLLIN|POLLHUP));
}
static bool read_exact(int fd,Bytes &b,unsigned timeout,bool eof_ok=false) {
    size_t offset=0;uint64_t deadline=monotonic_ns()+uint64_t(timeout)*1000000;
    while(offset<b.size()) {
        uint64_t now=monotonic_ns();
        if(now>=deadline||!readable(fd,unsigned((deadline-now+999999)/1000000)))throw std::runtime_error("socket idle timeout");
        ssize_t n=read(fd,b.data()+offset,b.size()-offset);
        if(n==0) {if(!offset&&eof_ok)return false;throw std::runtime_error("truncated socket request");}
        if(n<0) {if(errno==EINTR)continue;throw std::runtime_error("socket read");}
        offset+=size_t(n);
    }
    return true;
}
static void write_exact(int fd,const Bytes &b) {
    size_t offset=0;
    while(offset<b.size()) {
        ssize_t n=send(fd,b.data()+offset,b.size()-offset,MSG_NOSIGNAL);
        if(n<=0) {if(n<0&&errno==EINTR)continue;throw std::runtime_error("socket write");}
        offset+=size_t(n);
    }
}
static unsigned env_timeout() {
    const char *s=std::getenv("QUEST_PIPELINE_IDLE_MS");if(!s)return 30000;
    char *end=nullptr;unsigned long n=std::strtoul(s,&end,10);
    if(*end||n<100||n>300000)throw std::runtime_error("idle timeout range");
    return unsigned(n);
}
static void serve(Graph &backbone,Graph &fusion,const std::string &path,DetectorGraph *detector=nullptr) {
    if(path.empty()||path.size()>=sizeof(sockaddr_un::sun_path)||path[0]!='/') throw std::runtime_error("absolute socket path required");
    struct stat old{};if(lstat(path.c_str(),&old)==0)throw std::runtime_error("socket path already exists");
    if(errno!=ENOENT)throw std::runtime_error("socket path inspection");
    Descriptor listener(socket(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0));if(listener.fd<0)throw std::runtime_error("socket create");
    sockaddr_un address{};address.sun_family=AF_UNIX;std::strcpy(address.sun_path,path.c_str());
    if(bind(listener.fd,reinterpret_cast<sockaddr*>(&address),sizeof(address)))throw std::runtime_error("socket bind");
    SocketPath cleanup{path};if(chmod(path.c_str(),0600)||listen(listener.fd,1))throw std::runtime_error("socket listen");
    unsigned timeout=env_timeout();std::printf("stage=pipeline_ready socket=%s backbone_graph=%u fusion_graph=%u detector_graph=%u cpu_fallback=0\n",path.c_str(),backbone.gid,fusion.gid,detector?detector->gid:0);
    bool stop=false;uint64_t requests=0;
    while(!stop) {
        if(!readable(listener.fd,timeout))throw std::runtime_error("waiting client timeout");
        Descriptor connection(accept4(listener.fd,nullptr,nullptr,SOCK_CLOEXEC));if(connection.fd<0)throw std::runtime_error("socket accept");
        timeval tv{time_t(timeout/1000),suseconds_t((timeout%1000)*1000)};if(setsockopt(connection.fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof(tv)))throw std::runtime_error("socket send timeout");
        while(!stop) {
            Bytes header(32);if(!read_exact(connection.fd,header,timeout,true))break;
            uint64_t received=monotonic_ns();
            if(std::memcmp(header.data(),"QDP1",4))throw std::runtime_error("request magic");
            uint32_t op=at(header,4),size=at(header,24);uint64_t sequence=at64(header,8),exposure=at64(header,16);
            if(at(header,28)||op>3||(op==0&&size)||(op==1&&size!=backbone.input_bytes)||(op==2&&size!=fusion.input_bytes)||(op==3&&size!=DetectorGraph::input_bytes))throw std::runtime_error("request contract");
            if(op==3&&!detector)throw std::runtime_error("detector stage was not explicitly enabled");
            Bytes data(size);read_exact(connection.fd,data,timeout);
            Execution e;
            if(op==0)stop=true;
            else {e=op==3?detector->execute(data):(op==1?backbone:fusion).execute(data);++requests;}
            Bytes response{'Q','D','R','1'};word(response,op);word64(response,sequence);word64(response,exposure);word64(response,received);word64(response,e.start_ns);word64(response,e.end_ns);word64(response,monotonic_ns());word(response,0);word(response,uint32_t(e.data.size())+((op==1||op==2)?8:0));
            if(op==1||op==2) {
                uint32_t lo,hi;std::memcpy(&lo,&e.min,4);std::memcpy(&hi,&e.max,4);word(response,lo);word(response,hi);
            }
            response.insert(response.end(),e.data.begin(),e.data.end());
            write_exact(connection.fd,response);
            if(op)std::printf("stage=pipeline_execute op=%u sequence=%llu exposure_ns=%llu start_ns=%llu end_ns=%llu host_rpc_ms=%.6f bytes=%u\n",op,(unsigned long long)sequence,(unsigned long long)exposure,(unsigned long long)e.start_ns,(unsigned long long)e.end_ns,e.rpc_ms,unsigned(e.data.size()));
        }
    }
    std::printf("stage=pipeline_complete requests=%llu cpu_fallback=0\n",(unsigned long long)requests);
#ifdef XR_QUEST_DIAGNOSTIC_INNER_LOG
    backbone.m.dump_log();
#endif
}
extern "C" int a081_qvr_client_main() {
    std::setvbuf(stdout,nullptr,_IOLBF,0);std::setvbuf(stderr,nullptr,_IOLBF,0);
    try {
        const char *fdstr=std::getenv("HEXAGONRPC_FD"),*bundle=std::getenv("QUEST_HAND_BUNDLE"),*path=std::getenv("QUEST_PIPELINE_SOCKET");
        if(!fdstr||!bundle||!path)throw std::runtime_error("explicit fd, bundle, socket required");
        char *end=nullptr;long fd=std::strtol(fdstr,&end,10);
        if(*end||fd<0||fd>std::numeric_limits<int>::max()||fcntl(int(fd),F_GETFD)<0)throw std::runtime_error("inherited fd");
        Module m{int(fd)};m.open();m.init();m.build(load(std::string(bundle)+"/backbone-hvx.qhf",8*1024*1024));
        Graph backbone(m,{4,96,96,1},{4,4,4,60});
        m.init();m.build(load(std::string(bundle)+"/fusion-hvx.qhf",1024*1024));
        Graph fusion(m,{2,4,4,120},{2,4,4,60});
        std::unique_ptr<DetectorGraph> detector;
        const char *enable=std::getenv("QUEST_PIPELINE_DETECTOR");
        if(enable&&std::strcmp(enable,"0")) {
            if(std::strcmp(enable,"1"))throw std::runtime_error("detector enable must be0 or1");
            std::puts("stage=detector_prepare_begin validation=unverified");
            m.init();m.build(load(std::string(bundle)+"/detector-hvx.qhf",8*1024*1024));
            detector=std::make_unique<DetectorGraph>(m);
            std::printf("stage=detector_prepare_complete graph=%u rc=0\n",detector->gid);
        }
        serve(backbone,fusion,path,detector.get());return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"stage=pipeline_failed detail=%s\n",e.what());return 1;}
}
