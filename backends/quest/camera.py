#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""MIX shared Y8 frames and OEM fisheye -> virtual pinhole crops.

Consumes an existing owner's optional second FrameExporter endpoint. Never
opens V4L2/GPIO/IMU. All length/time/geometry failures are explicit; caller must
supply observed hand ROI or decoded hand landmarks, not a fabricated pose.
Calibration remains in the caller's private XML and is not embedded here.
"""
from __future__ import annotations
import array
from dataclasses import dataclass
import fcntl
import math
import os
from pathlib import Path
import socket
import struct
import time
import xml.etree.ElementTree as ET
import numpy as np

NAMES=('trackingA','trackingB','ctrl-trackingA','ctrl-trackingB')


def rigid_inverse(t):
    out=np.eye(4);out[:3,:3]=t[:3,:3].T;out[:3,3]=-out[:3,:3]@t[:3,3];return out


def rodrigues(v):
    v=np.asarray(v,float);theta=np.linalg.norm(v)
    if theta<1e-12:return np.eye(3)
    x,y,z=v/theta;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(theta)*k+(1-math.cos(theta))*(k@k)


def quaternion_transform(position,xyzw):
    q=np.asarray(xyzw,float);n=q@q
    if not np.isfinite(q).all() or abs(n-1)>.01:raise ValueError('Invalid measured head quaternion')
    x,y,z,w=q/np.sqrt(n)
    r=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)], [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)], [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=position;return t


@dataclass
class Camera:
    slot:int
    size:tuple
    focal:np.ndarray
    center:np.ndarray
    distortion:np.ndarray
    camera_from_reference:np.ndarray
    time_alignment_s:float

    def project(self,p):
        """OpenCV/KB4 fisheye theta polynomial; source requires forward rays."""
        p=np.asarray(p,float);radius=np.linalg.norm(p[...,:2],axis=-1);theta=np.arctan2(radius,p[...,2]);q=theta*theta
        k=self.distortion;d=theta*(1+q*(k[0]+q*(k[1]+q*(k[2]+q*k[3]))))
        ratio=np.divide(d,radius,out=np.ones_like(d),where=radius>1e-14)
        return p[...,:2]*ratio[...,None]*self.focal+self.center

    def unproject(self,p):
        uv=(np.asarray(p,float)-self.center)/self.focal;r=np.linalg.norm(uv,axis=-1);theta=r.copy();k=self.distortion
        for _ in range(12):
            q=theta*theta;f=theta*(1+q*(k[0]+q*(k[1]+q*(k[2]+q*k[3]))))-r
            df=1+q*(3*k[0]+q*(5*k[1]+q*(7*k[2]+q*9*k[3])))
            theta-=np.divide(f,df,out=np.zeros_like(f),where=abs(df)>1e-12)
        s=np.divide(np.sin(theta),r,out=np.ones_like(r),where=r>1e-14)
        return np.concatenate([uv*s[...,None],np.cos(theta)[...,None]],axis=-1)


@dataclass
class Calibration:
    cameras:list[Camera]
    body_from_reference:np.ndarray

    @classmethod
    def load(cls,path,*,rig_convention):
        if rig_convention!='rig-to-camera':raise ValueError('Explicit verified XML Rig direction is required')
        root=ET.parse(path).getroot();cams=[]
        def vec(node,key,n):
            x=np.fromstring(node.attrib[key],sep=' ')
            if x.size!=n or not np.isfinite(x).all():raise ValueError(key)
            return x
        for slot,name in enumerate(NAMES):
            cam=next(x for x in root.findall('Camera') if x.get('name')==name or x.get('cam_name')==name);c=cam.find('Calibration');r=cam.find('Rig')
            if c.get('model')!='FISHEYE_4_PARAMETERS':raise ValueError('Unsupported actual camera model')
            size=tuple(vec(c,'size',2).astype(int))
            if size!=(640,480):raise ValueError('Full resolution source and calibration must match')
            d=np.fromstring(c.attrib['radial_distortion'],sep=' ')
            if len(d)<4 or np.any(d[4:]):raise ValueError('Unsupported fisheye coefficient layout')
            t=np.eye(4);t[:3,:3]=vec(r,'rowMajorRotationMat',9).reshape(3,3);t[:3,3]=vec(r,'translation',3)
            if not np.allclose(t[:3,:3]@t[:3,:3].T,np.eye(3),atol=2e-5) or abs(np.linalg.det(t[:3,:3])-1)>2e-5:raise ValueError('Nonrigid OEM transform')
            cams.append(Camera(slot,size,vec(c,'focal_length',2),vec(c,'principal_point',2),d[:4],t,float(cam.find('TimeAlignment').attrib['delta'])))
        state=root.find('SFConfig/Stateinit');body=np.eye(4);body[:3,:3]=rodrigues(vec(state,'ombc',3));body[:3,3]=vec(state,'tbc',3)
        return cls(cams,body)


@dataclass
class Frame:
    images:list[np.ndarray]
    sequence:int
    source_frame_ids:tuple[int,int]
    sent_monotonic_ns:int
    received_monotonic_ns:int
    sof_boottime_ns:tuple[int,int]
    exposure_start_boottime_ns:tuple[int,int]
    exposure_center_boottime_ns:tuple[int,int]
    exposure_ns:int
    exposure_phase:int
    head_pose_boottime_ns:int
    world_from_body:np.ndarray
    camera_clock_bracket_ns:int
    producer_pid:int


class FrameReceiver:
    """Listener must exist before second exporter connects. No socket takeover."""
    def __init__(self,path):
        self.path=Path(path);self.client=None;self.last_sequence=0
        if os.geteuid()!=0:raise PermissionError('Exporter authenticates a root-only listener')
        if self.path.exists():raise FileExistsError('Never replace another observer endpoint')
        self.path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        st=self.path.parent.stat()
        if st.st_uid!=0 or st.st_mode&0o077:raise PermissionError('Private parent directory required')
        self.server=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET|socket.SOCK_CLOEXEC);self.server.bind(str(self.path));os.chmod(self.path,0o600);self.server.listen(1)

    def close(self):
        if self.client:self.client.close();self.client=None
        self.server.close();self.path.unlink(missing_ok=True)

    def receive(self,timeout=2):
        if self.client is None:
            self.server.settimeout(timeout);self.client,_=self.server.accept();self.client.settimeout(timeout)
            pid,uid,gid=struct.unpack('3i',self.client.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
            if uid!=0:self.client.close();self.client=None;raise PermissionError('Unexpected producer uid')
            self.last_sequence=0;self.producer_pid=pid
        fds=[]
        try:
            raw,anc,flags,_=self.client.recvmsg(216,socket.CMSG_SPACE(16),socket.MSG_CMSG_CLOEXEC);receive=time.monotonic_ns()
            for level,kind,payload in anc:
                if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:
                    items=array.array('i');items.frombytes(payload[:len(payload)//items.itemsize*items.itemsize]);fds.extend(items)
            if not raw:self.client.close();self.client=None;raise EOFError('Camera owner disconnected')
            if flags&(socket.MSG_TRUNC|socket.MSG_CTRUNC) or len(raw)!=216 or len(fds)!=1:raise ValueError('Truncated/missing shared frame')
            magic,version,size,kind,seq,sent=struct.unpack_from('<IIIIQQ',raw)
            if (magic,version,size,kind)!=(0x43494e41,3,216,2) or seq<=self.last_sequence:raise ValueError('Bad frame ABI/sequence')
            if not 0<=receive-sent<=100_000_000:raise ValueError('Stale shared frame')
            width,height,stride,frame0,frame1,valid=struct.unpack_from('<6I',raw,32)
            if (width,height,stride)!=(640,480,1280) or valid&15!=15:raise ValueError('Image/QCFT/head pose validity missing')
            sof0,sof1,start1,exposure,head_ns,bracket=struct.unpack_from('<6Q',raw,56);phase=struct.unpack_from('<I',raw,188)[0]
            if phase!=2:raise ValueError('Hand input needs the scheduled long-exposure phase')
            if not 0<exposure<1_000_000_000 or abs(sof0-sof1)>1_000_000 or bracket>=5000:raise ValueError('Unmatched camera timing')
            if start1!=sof1-exposure-19335 or not 0<=start1-head_ns<20_000_000:raise ValueError('Inconsistent exposure/pose timestamps')
            lengths=struct.unpack_from('<2I',raw,160);offsets=struct.unpack_from('<2I',raw,168)
            if lengths!=(614400,614400) or offsets!=(0,614400) or os.fstat(fds[0]).st_size!=1228800:raise ValueError('Invalid mosaic memory layout')
            required=fcntl.F_SEAL_WRITE|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_SEAL
            if fcntl.fcntl(fds[0],fcntl.F_GET_SEALS)&required!=required:raise ValueError('Expected immutable pixels')
            data=os.pread(fds[0],1228800,0)
            if len(data)!=1228800:raise ValueError('Short memfd read')
            mosaics=np.frombuffer(data,np.uint8).reshape(2,480,1280);images=[mosaics[i,:,:640].copy() if j==0 else mosaics[i,:,640:].copy() for i in range(2) for j in range(2)]
            position=np.array(struct.unpack_from('<3d',raw,104));quat=struct.unpack_from('<4d',raw,128)
            start0=sof0-exposure-19335;self.last_sequence=seq
            return Frame(images,seq,(frame0,frame1),sent,receive,(sof0,sof1),(start0,start1),(start0+exposure//2,start1+exposure//2),exposure,phase,head_ns,quaternion_transform(position,quat),bracket,self.producer_pid)
        finally:
            for fd in fds:os.close(fd)


def crop_from_landmarks(camera,points_reference,*,right_hand=False,camera_angle_deg=0,size=96,focal_multiplier=.95):
    """Generate UmeTrack-style perspective ROI from real 3D landmarks.

    Coordinates and camera translation use metres. Right-hand reflection is
    applied to the virtual camera. No learned model or fake initializer here.
    """
    points=np.asarray(points_reference,float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<2 or not np.isfinite(points).all():raise ValueError('Measured 3D crop support required')
    center=(points.min(axis=0)+points.max(axis=0))/2;t=camera.camera_from_reference;eye=t[:3,:3]@center+t[:3,3];n=np.linalg.norm(eye)
    if n<1e-5:raise ValueError('Invalid hand center')
    direction=eye/n;axis=np.cross([0.,0.,1.],direction);cross=np.linalg.norm(axis);dot=direction[2]
    if dot<=-1+1e-8:raise ValueError('Backward-facing crop')
    delta=rodrigues(axis/cross*math.atan2(cross,dot)) if cross>1e-12 else np.eye(3)
    reference_from_crop=rigid_inverse(t);reference_from_crop[:3,:3]=reference_from_crop[:3,:3]@delta@rodrigues([0,0,math.radians(camera_angle_deg)])
    crop_from_reference=rigid_inverse(reference_from_crop)
    if right_hand:crop_from_reference[0,:]*=-1
    transformed=points@crop_from_reference[:3,:3].T+crop_from_reference[:3,3]
    if np.min(transformed[:,2])<1e-4:raise ValueError('Crop support behind camera')
    c=(size-1)/2;f=focal_multiplier*c/np.max(abs(transformed[:,:2]/transformed[:,2:]))
    if not np.isfinite(f) or f<5:raise ValueError('Degenerate crop focal')
    return np.array([[f,0,c],[0,f,c],[0,0,1.]],np.float32),crop_from_reference.astype(np.float32)



def crop_from_bbox(camera,bbox,*,right_hand=False,camera_angle_deg=0,size=96,focal_multiplier=.95):
    """Virtual-camera ROI from an observed distorted-image box.

    The box defines angular image support only. It neither estimates a hand
    depth nor supplies a metric hand pose to the network or tracker state.
    """
    b=np.asarray(bbox,float)
    if b.shape!=(4,) or not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1]:raise ValueError('Invalid observed detector bbox')
    x0,y0,x1,y1=b;corners=np.array([[x0,y0],[x1,y0],[x0,y1],[x1,y1]])
    direction=camera.unproject(np.array([(x0+x1)/2,(y0+y1)/2]));axis=np.cross([0.,0.,1.],direction);cross=np.linalg.norm(axis);dot=direction[2]
    if dot<=-1+1e-8:raise ValueError('Backward-facing detector ROI')
    delta=rodrigues(axis/cross*math.atan2(cross,dot)) if cross>1e-12 else np.eye(3)
    reference_from_crop=rigid_inverse(camera.camera_from_reference);reference_from_crop[:3,:3]=reference_from_crop[:3,:3]@delta@rodrigues([0,0,math.radians(camera_angle_deg)])
    crop_from_reference=rigid_inverse(reference_from_crop)
    if right_hand:crop_from_reference[0,:]*=-1
    # Rotation between ray frames; camera centres coincide, so no translation.
    local_rays=camera.unproject(corners)@camera.camera_from_reference[:3,:3]@crop_from_reference[:3,:3].T
    if np.min(local_rays[:,2])<1e-4:raise ValueError('Detector ROI outside forward hemisphere')
    c=(size-1)/2;extent=np.max(abs(local_rays[:,:2]/local_rays[:,2:]));f=focal_multiplier*c/extent
    if not np.isfinite(f) or f<5:raise ValueError('Degenerate detector crop')
    return np.array([[f,0,c],[0,f,c],[0,0,1.]],np.float32),crop_from_reference.astype(np.float32)


def warp_crop(image,camera,intrinsics,crop_from_reference,*,size=96):
    """CPU geometry preprocessing only; returns original UINT8 model input."""
    image=np.asarray(image)
    if image.dtype!=np.uint8 or image.shape!=(camera.size[1],camera.size[0]):raise ValueError('Image must match actual OEM calibration resolution')
    yy,xx=np.mgrid[:size,:size];p=np.stack([xx,yy,np.ones_like(xx)],axis=-1).reshape(-1,3)@np.linalg.inv(intrinsics).T
    # Virtual/source camera centres coincide. Reflections are orthonormal.
    rays=p@crop_from_reference[:3,:3]@camera.camera_from_reference[:3,:3].T
    q=camera.project(rays);valid=(rays[:,2]>0)&np.isfinite(q).all(axis=1);x=np.where(valid,q[:,0],-10);y=np.where(valid,q[:,1],-10)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);dx=x-x0;dy=y-y0;out=np.zeros(len(x),float)
    for ox,oy,w in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        sx=x0+ox;sy=y0+oy;ok=(sx>=0)&(sx<image.shape[1])&(sy>=0)&(sy<image.shape[0])&valid
        out[ok]+=image[sy[ok],sx[ok]]*w[ok]
    return np.clip(np.floor(out+.5),0,255).astype(np.uint8).reshape(size,size)


def prepare_crops(frame,calibration,rois):
    """rois=[{hand:0|1,slot:0..3,points_reference:Nx3,angle_deg?:...},...]."""
    if len(rois)!=4:raise ValueError('This recovered DSP graph requires four real crops (two views per hand)')
    records=[];crops=[]
    for hand in (0,1):
        hand_rois=[r for r in rois if r['hand']==hand]
        if len(hand_rois)!=2 or len({r['slot'] for r in hand_rois})!=2:raise ValueError('Two distinct real camera views are required per hand')
        for roi in sorted(hand_rois,key=lambda r:r['slot']):
            cam=calibration.cameras[roi['slot']]
            generate=crop_from_bbox if 'bbox' in roi else crop_from_landmarks
            support=roi['bbox'] if 'bbox' in roi else roi['points_reference']
            k,ext=generate(cam,support,right_hand=hand==1,camera_angle_deg=roi.get('angle_deg',0))
            crops.append(warp_crop(frame.images[cam.slot],cam,k,ext));records.append({'hand':hand,'slot':cam.slot,'intrinsics':k.tolist(),'crop_from_reference':ext.tolist()})
    return np.stack(crops)[...,None],records


def model_camera_geometry(frame,calibration,crop_records):
    """Compose raw-QVR measured body pose with OEM camera/body calibration.

    Preserves actual prior pose timestamp rather than predicting to the image.
    Consumer must retain head_pose_boottime_ns and associated frame times.
    Returned extrinsics are world->virtual-camera in metres, not XR eye poses.
    """
    world_from_reference=frame.world_from_body@calibration.body_from_reference
    reference_from_world=rigid_inverse(world_from_reference)
    intrinsics=np.array([r['intrinsics'] for r in crop_records],np.float32)
    extrinsics=np.stack([np.asarray(r['crop_from_reference'])@reference_from_world for r in crop_records]).astype(np.float32)
    return intrinsics,extrinsics
