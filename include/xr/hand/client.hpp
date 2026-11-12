// SPDX-License-Identifier: MIT
#pragma once
#include <xr/hand/state.h>
#include <cerrno>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <cmath>
#include <cstring>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

namespace xr::hand {
class Client {
public:
    ~Client() { reset(); }
    bool configure(const char *path) {
        if (!path || !*path) return false;
        if (path[0] != '/' || std::strlen(path) >= sizeof(path_) || std::strstr(path, "/../")) return false;
        std::strcpy(path_, path); return true;
    }
    static uint64_t now() {
        timespec t{}; if (clock_gettime(CLOCK_BOOTTIME, &t)) return 0;
        return uint64_t(t.tv_sec)*1000000000ULL + t.tv_nsec;
    }
    void reset() { if (fd_ >= 0) close(fd_); fd_ = -1; { std::lock_guard<std::mutex> lock(mutex_); packet_ = {}; } }
    void update() {
        const auto t = now();
        if (fd_ < 0 && t >= retry_) { retry_ = t + 1000000000ULL; connect_source(); }
        for (unsigned i=0; fd_>=0 && i<64; ++i) {
            xr_hand_packet p{};
            auto n=recv(fd_, &p, sizeof(p), MSG_DONTWAIT | MSG_TRUNC);
            if (n<0 && (errno==EAGAIN || errno==EWOULDBLOCK || errno==EINTR)) break;
            if (n!=sizeof(p) || !accept(p,now())) { reset(); break; }
            { std::lock_guard<std::mutex> lock(mutex_); packet_=p; }
        }
    }
    std::optional<xr_hand_state> get(unsigned hand, xr_hand_packet *snapshot=nullptr) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto t=now();
        if (snapshot) *snapshot=packet_;
        if (hand>1 || packet_.source!=XR_HAND_SOURCE_LIVE || !packet_.sequence ||
            packet_.decoded_boottime_ns>t || t-packet_.decoded_boottime_ns>250000000ULL ||
            packet_.exposure_boottime_ns>t || t>=packet_.valid_until_boottime_ns ||
            !(packet_.hands[hand].flags & XR_HAND_VALID)) return std::nullopt;
        return packet_.hands[hand];
    }
    xr_hand_packet packet() const { std::lock_guard<std::mutex> lock(mutex_); return packet_; }
private:
    bool accept(const xr_hand_packet &p,uint64_t t) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (p.magic!=XR_HAND_MAGIC || p.version!=XR_HAND_VERSION || p.bytes!=sizeof(p) ||
            p.source!=XR_HAND_SOURCE_LIVE || p.reserved || !p.producer_pid || p.decoded_boottime_ns>t ||
            !p.exposure_boottime_ns || p.valid_until_boottime_ns<p.exposure_boottime_ns ||
            p.valid_until_boottime_ns-p.exposure_boottime_ns>250000000ULL ||
            (p.producer_pid==packet_.producer_pid && p.sequence<packet_.sequence)) return false;
        for (unsigned i=0;i<2;++i) {
            auto &h=p.hands[i]; if (h.hand!=i || (h.flags&~7u)) return false;
            if (!(h.flags&XR_HAND_VALID)) { if (h.flags || h.pinch || h.grab) return false; continue; }
            if (!(h.pinch>=0 && h.pinch<=1 && h.grab>=0 && h.grab<=1)) return false;
            for (const float *pose : {h.grip,h.aim}) {
                float norm=0;
                for (unsigned j=0;j<7;++j) if (!std::isfinite(pose[j])) return false;
                for (unsigned j=3;j<7;++j) norm+=pose[j]*pose[j];
                if (std::fabs(norm-1.f)>.01f) return false;
            }
            for (auto &lm:h.landmarks) for (float x:lm) if (!std::isfinite(x)) return false;
        }
        return true;
    }
    void connect_source() {
        struct stat st{};
        if (lstat(path_,&st) || !S_ISSOCK(st.st_mode) || st.st_uid!=0) return;
        int candidate=socket(AF_UNIX,SOCK_SEQPACKET|SOCK_NONBLOCK|SOCK_CLOEXEC,0);
        if (candidate<0) return;
        sockaddr_un address{};address.sun_family=AF_UNIX;std::strcpy(address.sun_path,path_);
        ucred peer{};socklen_t bytes=sizeof(peer);
        if (connect(candidate,reinterpret_cast<sockaddr*>(&address),sizeof(address)) ||
            getsockopt(candidate,SOL_SOCKET,SO_PEERCRED,&peer,&bytes) || bytes!=sizeof(peer) || peer.uid!=0) {close(candidate);return;}
        fd_=candidate;
    }
    mutable std::mutex mutex_;
    xr_hand_packet packet_{};
    int fd_{-1}; uint64_t retry_{};
    char path_[sizeof(sockaddr_un::sun_path)]{};
};
} // namespace xr::hand
