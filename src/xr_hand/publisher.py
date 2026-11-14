# SPDX-License-Identifier: MIT
"""Measured 6DoF hand poses and hysteretic pinch/grab publication.

Aim follows the palm axis independently of finger flexion; grip uses the palm. Both are defined
by the decoded hand geometry and share the head tracker's metric world.
"""
import os,socket,threading,time
from pathlib import Path
import numpy as np
from .protocol import HEADER,HAND,MAGIC,SIZE
B=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])

def quaternion(r):
    # Symmetric eigensystem avoids branch singularities near 180 degree turns.
    a=r
    k=np.array([[a[0,0]-a[1,1]-a[2,2],a[0,1]+a[1,0],a[0,2]+a[2,0],a[2,1]-a[1,2]],
                [a[0,1]+a[1,0],a[1,1]-a[0,0]-a[2,2],a[1,2]+a[2,1],a[0,2]-a[2,0]],
                [a[0,2]+a[2,0],a[1,2]+a[2,1],a[2,2]-a[0,0]-a[1,1],a[1,0]-a[0,1]],
                [a[2,1]-a[1,2],a[0,2]-a[2,0],a[1,0]-a[0,1],np.trace(a)]])/3
    _,v=np.linalg.eigh(k);q=v[:,-1]
    return q if q[3]>=0 else -q

def pose(origin,direction,normal):
    z=-direction/np.linalg.norm(direction);x=np.cross(normal,z);n=np.linalg.norm(x)
    if n<1e-5:raise ValueError('Degenerate measured hand basis')
    x/=n;y=np.cross(z,x);r=np.column_stack((x,y,z))
    return np.r_[origin,quaternion(r)]

def measure_hand(landmarks,hand):
    """Measure poses and gesture scores without asserting source validity."""
    if hand not in (0,1):raise ValueError("Unknown hand slot")
    lm=np.asarray(landmarks,float)@B.T
    if lm.shape!=(21,3) or not np.isfinite(lm).all():raise ValueError('Nonfinite hand geometry')
    palm=lm[20];normal=np.cross(lm[8]-lm[5],lm[17]-lm[5])*(-1 if hand else 1)
    grip=pose(palm,lm[11]-lm[5],normal)
    # MCPs and wrist stay stable when the thumb/index tips close.
    aim=pose(np.mean(lm[[5,8,11,14,17]],axis=0),lm[11]-lm[5],normal)
    width=np.linalg.norm(lm[8]-lm[17])
    if not .025<width<.15:raise ValueError('Invalid decoded palm width')
    pinch=float(np.clip((.55-np.linalg.norm(lm[0]-lm[1])/width)/.4,0,1))
    curls=[]
    for tip,mcp,pip,dip in [(1,8,9,10),(2,11,12,13),(3,14,15,16),(4,17,18,19)]:
        length=np.linalg.norm(lm[pip]-lm[mcp])+np.linalg.norm(lm[dip]-lm[pip])+np.linalg.norm(lm[tip]-lm[dip])
        if length<=.005:raise ValueError('Degenerate finger chain')
        curls.append(1-np.linalg.norm(lm[tip]-lm[mcp])/length)
    grab=float(np.clip(np.median(curls)/.55,0,1))
    return grip,aim,pinch,grab,lm

class AimFilter:
    """Palm ray smoothing with a small, continuous pinch motion deadband.

    Freeze the last relaxed aim while fingers close. Deliberate palm motion
    smoothly escapes the deadband, so a held pinch can still drag. Never keep
    history across tracking loss, repeated sequence or an exposure gap.
    """
    def __init__(self):
        self.reset()
    def reset(self):
        self.filtered=None;self.output=None;self.anchor=None;self.time=0
    @staticmethod
    def blend(a,b,t):
        q=b[3:].copy()
        if np.dot(a[3:],q)<0:q=-q
        q=(1-t)*a[3:]+t*q;q/=np.linalg.norm(q)
        return np.r_[(1-t)*a[:3]+t*b[:3],q]
    def update(self,aim,pinch,stamp):
        dt=(stamp-self.time)/1e9
        if self.filtered is None or not 0<dt<=.25:
            self.reset();self.filtered=aim.copy();self.output=aim.copy();self.time=stamp
            return aim.copy()
        delta=np.linalg.norm(aim[:3]-self.filtered[:3])
        angle=2*np.arccos(np.clip(abs(np.dot(aim[3:],self.filtered[3:])),0,1))
        cutoff=4+min(16,delta/dt*20+angle/dt*2)
        self.filtered=self.blend(self.filtered,aim,1-np.exp(-2*np.pi*cutoff*dt))
        if self.anchor is None and pinch>=.35:self.anchor=self.output.copy()
        if self.anchor is not None and pinch<=.25:
            # Release from the displayed ray, avoiding a one-frame snap.
            self.filtered=self.blend(self.output,self.filtered,1-np.exp(-2*np.pi*4*dt))
            self.anchor=None
        out=self.filtered.copy()
        if self.anchor is not None:
            distance=np.linalg.norm(out[:3]-self.anchor[:3])/.012
            rotation=2*np.arccos(np.clip(abs(np.dot(out[3:],self.anchor[3:])),0,1))/np.deg2rad(3)
            t=float(np.clip(max(distance,rotation)-1,0,1));t=t*t*(3-2*t)
            out=self.blend(self.anchor,out,t)
        self.output=out;self.time=stamp
        return out.copy()

class Publisher:
    def __init__(self,path,source=1):
        if source not in (1,2):raise ValueError('Invalid publication source')
        self.path=Path(path);self.source=source;self.clients=[];self.latest=None;self.lock=threading.Lock();self.closed=threading.Event();self.latched=np.zeros((2,2),bool);self.last_sequence=0;self.producer=None;self.aim_filters=[AimFilter(),AimFilter()]
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists():raise FileExistsError('Another hand publisher owns endpoint')
        self.server=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET|socket.SOCK_CLOEXEC)
        self.server.bind(str(self.path));os.chmod(self.path,0o666);self.server.listen(8);self.server.settimeout(.2)
        self.thread=threading.Thread(target=self.accept,daemon=True);self.thread.start()
    def accept(self):
        while not self.closed.is_set():
            try:c,_=self.server.accept()
            except (TimeoutError,socket.timeout):continue
            except OSError:break
            c.setblocking(False)
            with self.lock:
                if len(self.clients)>=16:c.close();continue
                if self.latest:
                    try:c.send(self.latest)
                    except OSError:c.close();continue
                self.clients.append(c)
    def publish(self,record):
        now=time.clock_gettime_ns(time.CLOCK_BOOTTIME);sequence=record.get('sequence',self.last_sequence)
        exposure=min(record.get('exposure_start_boottime_ns',[now]));until=record.get('valid_until_boottime_ns',now)
        if sequence<=self.last_sequence or self.producer!=record.get('producer_pid'):
            self.latched[:]=False
            for f in self.aim_filters:f.reset()
        self.last_sequence=sequence;self.producer=record.get('producer_pid');entries=[]
        for hand in (0,1):
            item=next((x for x in record.get('hands',[]) if x.get('hand')==hand),{})
            try:
                if not item.get('valid') or not exposure<=now<until:raise ValueError('Invalid or expired hand')
                grip,aim,pinch,grab,lm=measure_hand(item['landmarks_world_m'],hand)
                aim=self.aim_filters[hand].update(aim,pinch,exposure)
                values=[pinch,grab]
                for j,v in enumerate(values):self.latched[hand,j]=v>=(.35 if self.latched[hand,j] else .65)
                flags=1|int(self.latched[hand,0])*2|int(self.latched[hand,1])*4
                entries.append(HAND.pack(hand,flags,*grip,*aim,pinch,grab,*lm.ravel()))
            except (ValueError,KeyError,np.linalg.LinAlgError):
                self.latched[hand]=False;self.aim_filters[hand].reset();entries.append(HAND.pack(hand,0,*([0.]*79)))
        packet=HEADER.pack(MAGIC,1,SIZE,self.source,sequence,exposure,now,until,os.getpid(),0)+b''.join(entries)
        with self.lock:
            self.latest=packet;kept=[]
            for c in self.clients:
                try:
                    if c.send(packet)!=SIZE:raise OSError('Short publication')
                    kept.append(c)
                except OSError:c.close()
            self.clients=kept
    def close(self):
        self.publish({'kind':'stopped','sequence':self.last_sequence,'hands':[]})
        self.closed.set();self.server.close();self.thread.join(timeout=1)
        with self.lock:
            for c in self.clients:c.close()
        self.path.unlink(missing_ok=True)
