#!/usr/bin/env python3
"""Original-weight DPEV30MG host-side stages around the two DSP delegates.

Requires Python 3 + NumPy only. These FP32 stages are the original serialized
XNNPACK stages; they do not replace either neural DSP subgraph. FTL and external
SVD/skinning formulas are adapted from Meta UmeTrack (CC BY-NC 4.0), commit
 e263ec0844abf73457468a050a75853444d33d69. Their equivalence to the private native
Quest operators remains unverified. Original binary static inspection recovered
the 200-pixel focal normalization and the FTL affine/channel-block direction;
see native-geometry-recovery.json. All spatial values are meters.
"""
from pathlib import Path
import json
import numpy as np

BACKBONE_SCALE=np.float32(0.008666601032018661)
FUSION_INPUT_SCALE=np.float32(0.006989060435444117)
FUSION_OUTPUT_SCALE=np.float32(0.006207913625985384)


def model_geometry(intrinsics,world_to_left_crop,previous_world_to_camera0=None,use_memory=None,canonical_focal_length=200.0):
    """Build all FTL transforms from real virtual-crop intrinsics/extrinsics.

    Canonical focal 200 is independently present in the original Quest binary's
    registered ComputeSingleViewTransformsCPUKernel: float32 0x43480000 at
    0x1e1a5cc, intrinsics[0,0]/200 at 0x1e1a630..0x1e1a648, stored at matrix[2,2].
    This is static instruction/dataflow evidence, not original-library numerical
    execution. The full multi-view factorization follows public UmeTrack and
    serialized `_use_unscaled_as_canonical=False`; its entire private-native
    multiplication order was not independently recovered in this bounded pass.

    E is world->cropped left-canonical camera, translations in METERS. Right
    crops include a reflected X camera axis in E. For each hand use E0 from the
    same exposure as its two image crops. Ordering: L0,L1,R0,R1.
    """
    k=np.asarray(intrinsics,np.float32);e=np.asarray(world_to_left_crop,np.float32)
    if k.shape!=(4,3,3) or e.shape!=(4,4,4):raise ValueError('Expected crop K[4,3,3], E[4,4,4]')
    if not np.isfinite(k).all() or not np.isfinite(e).all() or np.any(k[:,0,0]<=0) or canonical_focal_length<=0:raise ValueError('Invalid camera geometry')
    if not np.allclose(e[:,3],[0,0,0,1],atol=1e-5):raise ValueError('Extrinsics must be affine')
    if not np.allclose(k[:,0,0],k[:,1,1],rtol=1e-4):raise ValueError('FTL expects square-pixel virtual crop cameras')
    s=np.repeat(np.eye(4,dtype=np.float32)[None],4,0);s[:,2,2]=k[:,0,0]/np.float32(canonical_focal_length)
    ev=e.reshape(2,2,4,4);sv=s.reshape(2,2,4,4);canonical_to_cam0=sv[:,0].copy()
    canonical=np.linalg.inv(sv[:,0,None])@ev[:,0,None]@np.linalg.inv(ev)@sv
    memory_used=np.zeros(2,bool) if use_memory is None else np.asarray(use_memory,bool).reshape(2).copy()
    temporal=np.repeat(np.eye(4,dtype=np.float32)[None],2,0)
    if previous_world_to_camera0 is None:memory_used[:]=False
    elif memory_used.any():
        prev=np.asarray(previous_world_to_camera0,np.float32)
        if prev.shape!=(2,4,4):raise ValueError('Previous camera0 extrinsics must be 2x4x4')
        temporal[memory_used]=ev[memory_used,0]@np.linalg.inv(prev[memory_used])
    return {'T_camera_image':s,'T_canonical_image':canonical.reshape(4,4,4).astype(np.float32),'T_camera0_canonical':canonical_to_cam0,'T_curFrame_prevFrame':temporal,'world_to_camera0':ev[:,0].copy(),'T_world_left_camera0':np.linalg.inv(ev[:,0]).astype(np.float32),'use_memory':memory_used,'canonical_focal_length':float(canonical_focal_length),'canonical_focal_original_binary_verified':float(canonical_focal_length)==200.0,'ftl_affine_direction_original_binary_verified':True,'private_nimble_numeric_equivalence':False}


def prepare_temporal_memory(memory,geometry):
    """Explicit per-hand zero/reset before affine FTL; a translated zero tensor
    is not a zero tensor, so reset slots must use identity transforms as above.
    """
    m=np.asarray(memory,np.float32).copy()
    if m.shape!=(2,18,4,4):raise ValueError('Memory must be 2x18x4x4')
    m[~geometry['use_memory']]=0
    return m


def ftl(xfs,features,ratio=.5):
    """NCHW first C*ratio channels -> (N,3,-1), affine, then concatenate.

    Native static dataflow corroboration: original Quest kernel 0x1e1c514 uses
    round(C*ratio)/3*H*W points, preserves the remaining cloned feature tensor,
    and calls helper 0x1e2393c. That helper reads three contiguous point blocks
    and computes R[row,0]*X + R[row,1]*Y + R[row,2]*Z + T[row,3]. Exact private
    FMA/rounding numerical equivalence has not been executed or measured.
    """
    x=np.asarray(features,np.float32);t=np.asarray(xfs,np.float32)
    if x.ndim!=4 or t.shape!=(x.shape[0],4,4):raise ValueError('FTL requires NCHW and matching N44 matrices')
    if not np.isfinite(x).all() or not np.isfinite(t).all():raise ValueError('Nonfinite FTL input')
    if not 0<=ratio<=1:raise ValueError('FTL ratio must lie in [0,1]')
    # Original native FCVTAS rounds positive halves away from zero.
    nc=int(np.floor(x.shape[1]*ratio+.5))
    if nc%3:raise ValueError('FTL channels must be divisible by 3')
    if nc==0:return x.copy()
    y=(t[:,:3,:3]@x[:,:nc].reshape(len(x),3,-1)+t[:,:3,3,None]).reshape(x[:,:nc].shape)
    return np.ascontiguousarray(np.concatenate((y,x[:,nc:]),axis=1))


def sigmoid(x):
    x=np.asarray(x,np.float32);return np.exp(-np.logaddexp(np.float32(0),-x)).astype(np.float32)


def quantize_fusion(features):
    """NCHW float -> NHWC uint8, original calibrated range (scale,zp143)."""
    x=np.asarray(features,np.float32)
    if x.shape!=(2,120,4,4) or not np.isfinite(x).all():raise ValueError('Fusion input must be finite 2x120x4x4')
    q=np.clip(np.floor(x/FUSION_INPUT_SCALE+np.float32(143.5)),0,255).astype(np.uint8)
    return np.ascontiguousarray(q.transpose(0,2,3,1))


def dequantize(raw,shape,scale,zero):
    q=np.frombuffer(raw,np.uint8).reshape(shape) if isinstance(raw,(bytes,bytearray,memoryview)) else np.asarray(raw,np.uint8)
    if q.shape!=shape:raise ValueError((q.shape,shape))
    return np.ascontiguousarray(((q.astype(np.float32)-np.float32(zero))*np.float32(scale)).transpose(0,3,1,2))


def wrist_svd(raw_points,source_points):
    to=np.asarray(raw_points,np.float32).reshape(-1,7,3);src=np.broadcast_to(source_points,to.shape)
    sm=src.mean(1);tm=to.mean(1);cov=(src-sm[:,None]).transpose(0,2,1)@(to-tm[:,None])
    u,_,vh=np.linalg.svd(cov);v=vh.transpose(0,2,1);det=np.linalg.det(v@u.transpose(0,2,1))
    d=np.broadcast_to(np.eye(3,dtype=np.float32),(len(to),3,3)).copy();d[:,2,2]=det
    r=v@d@u.transpose(0,2,1);out=np.broadcast_to(np.eye(4,dtype=np.float32),(len(to),4,4)).copy()
    out[:,:3,:3]=r;out[:,:3,3]=tm-(r@sm[...,None])[...,0]
    return out


def joint_local_xfs(axes,positions,angles):
    """Rodrigues rotation around original joint axis passing through rest point."""
    aa=np.asarray(axes,np.float32)*np.asarray(angles,np.float32)[...,None];p=np.asarray(positions,np.float32)
    theta=np.linalg.norm(aa,axis=-1);k=np.zeros(aa.shape[:-1]+(3,3),np.float32)
    k[...,0,1]=-aa[...,2];k[...,0,2]=aa[...,1];k[...,1,0]=aa[...,2];k[...,1,2]=-aa[...,0];k[...,2,0]=-aa[...,1];k[...,2,1]=aa[...,0]
    a=np.sinc(theta/np.pi);b=np.float32(.5)*np.sinc(theta/(2*np.pi))**2
    r=np.eye(3,dtype=np.float32)+a[...,None,None]*k+b[...,None,None]*(k@k)
    out=np.broadcast_to(np.eye(4,dtype=np.float32),aa.shape[:-1]+(4,4)).copy();out[...,:3,:3]=r;out[...,:3,3]=p-(r@p[...,None])[...,0]
    return out


class DPEHostStages:
    def __init__(self,folder=None):
        folder=Path(folder) if folder else Path(__file__).resolve().parent
        self.meta=json.loads((folder/'contract.json').read_text());self.w=dict(np.load(folder/'original-decoder-weights.npz'))
        self.folder=folder;self._ort=None
    def enable_onnx(self,path=None,intra_op_threads=1):
        """Optional CPU optimization for original host stages; DSP graphs unchanged."""
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=intra_op_threads;options.inter_op_num_threads=1
        self._ort=ort.InferenceSession(str(path or self.folder/'original-host-stages.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    def linear(self,index,x):
        c=self.meta['constants']['c'+str(index)];y=np.asarray(x,np.float32)@self.w[c['weight']].T+self.w[c['bias']]
        return self.clamp(c,y)
    @staticmethod
    def clamp(c,y):
        if c['output_min'] is not None:y=np.maximum(y,np.float32(c['output_min']))
        if c['output_max'] is not None:y=np.minimum(y,np.float32(c['output_max']))
        return y
    def conv(self,index,x):
        c=self.meta['constants']['c'+str(index)];w=self.w[c['weight']];bias=self.w[c['bias']];x=np.asarray(x,np.float32)
        if c['groups']!=1:raise ValueError('Unexpected grouped conv')
        ph,pw=c['padding'];sh,sw=c['stride'];dh,dw=c['dilation'];kh,kw=w.shape[2:]
        x=np.pad(x,((0,0),(0,0),(ph,ph),(pw,pw))) if ph or pw else x
        win=np.lib.stride_tricks.sliding_window_view(x,((kh-1)*dh+1,(kw-1)*dw+1),axis=(2,3))[:,:,::sh,::sw,::dh,::dw]
        n,_,h,ww,_,_=win.shape
        col=np.ascontiguousarray(win.transpose(0,2,3,1,4,5)).reshape(n*h*ww,-1)
        y=(col@w.reshape(len(w),-1).T+bias).reshape(n,h,ww,len(w)).transpose(0,3,1,2)
        return np.ascontiguousarray(self.clamp(c,y))
    def prepare_fusion(self,backbone_u8,T_camera_image,T_canonical_image):
        """Order is hand0/view0, hand0/view1, hand1/view0, hand1/view1.

        Transforms describe the *cropped and left-canonicalized* input cameras.
        T_camera_image maps each virtual normalized image-space feature to its
        unscaled cropped-camera space. T_canonical_image maps each to the same
        hand-specific canonical camera0 coordinate system.
        """
        per=dequantize(backbone_u8,(4,4,4,60),BACKBONE_SCALE,143)
        visibility=sigmoid(self.linear(0,per.reshape(4,-1)))
        single=ftl(np.asarray(T_camera_image,np.float32)[::2],per[::2])
        multiv=ftl(T_canonical_image,per).reshape(2,120,4,4)
        return quantize_fusion(multiv),{'singlev_features':single,'encoder_visibility_prediction':visibility}
    def skeleton_features(self,axes=None,positions=None):
        axes=self.w['joint_rotation_axes'] if axes is None else np.asarray(axes,np.float32)
        positions=self.w['joint_rest_positions'] if positions is None else np.asarray(positions,np.float32)
        if axes.ndim==2:axes=np.broadcast_to(axes,(2,22,3))
        if positions.ndim==2:positions=np.broadcast_to(positions,(2,22,3))
        if axes.shape!=(2,22,3) or positions.shape!=axes.shape:raise ValueError('Expected two left-canonical 22-joint skeletons in meters')
        y=self.linear(1,np.concatenate((axes,positions),axis=-1).reshape(2,132)).reshape(2,4,4,4)
        gamma,beta,mean,var=[self.w['c'+str(i)][None,:,None,None] for i in [2,3,4,5]]
        return np.maximum((y-mean)/np.sqrt(var+np.float32(1e-5))*gamma+beta,0).astype(np.float32)
    def pose_head(self,features,start):
        a=self.conv(start+1,self.conv(start,features));a=np.maximum(a+features,0)
        b=self.conv(start+3,self.conv(start+2,a));return self.conv(start+4,np.maximum(b+a,0)).mean((2,3))
    def finish(self,fusion_u8,encoder_context,T_camera0_canonical,valid_multi_view,T_curFrame_prevFrame,rnn_mem_features,axes=None,positions=None,do_shape_prediction=False):
        multi=ftl(T_camera0_canonical,dequantize(fusion_u8,(2,4,4,60),FUSION_OUTPUT_SCALE,110))
        valid=np.asarray(valid_multi_view,np.float32).reshape(2,1,1,1)
        if not np.isin(valid,[0,1]).all():raise ValueError('Multi-view validity must be binary')
        img=encoder_context['singlev_features']*(1-valid)+multi*valid
        mem=np.asarray(rnn_mem_features,np.float32)
        if mem.shape!=(2,18,4,4):raise ValueError('RNN memory must be 2x18x4x4')
        transformed_mem=ftl(T_curFrame_prevFrame,mem);skel=self.skeleton_features(axes,positions)
        if self._ort is None:
            temporal=self.conv(8,self.conv(7,self.conv(6,np.concatenate((transformed_mem,img),axis=1))))
            raw=self.pose_head(np.concatenate((temporal[:,18:],skel),axis=1),9)
            mem_out=temporal[:,:18].copy();features=self.conv(16,img);existence=sigmoid(self.conv(15,self.conv(14,img)).mean((2,3))[:,0])
            scale=self.pose_head(temporal[:,18:],17)[:,0].copy() if do_shape_prediction else None
        else:
            raw,mem_out,existence,features,scale=self._ort.run(None,{'image':np.ascontiguousarray(img),'transformed_memory':transformed_mem,'skeleton_features':skel})
        result={'joint_angles':raw[:,:20].copy(),'wrist_xfs':raw[:,20:41].copy(),'landmark_uncertainty_sigmas':raw[:,41:62].copy(),'rnn_mem_features':mem_out,'encoder_visibility_prediction':encoder_context['encoder_visibility_prediction'],'mv_img_features':features,'hand_existence':existence}
        if do_shape_prediction:result['skel_scales']=scale
        if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite original-weight head output')
        return result
    def decode(self,raw):
        """Decode raw network outputs in cropped camera0/left-canonical coordinates.

        Do not publish as a tracked hand without caller's detection, reprojection,
        confidence and temporal validity checks. sigma scales from the private
        native runtime are not applied here because that policy is unverified.
        """
        angles=np.concatenate((raw['joint_angles'],np.zeros((2,2),np.float32)),axis=1)
        wrist=wrist_svd(raw['wrist_xfs'],self.w['_left_wrist_sample_points'])
        result={'joint_angles':angles,'wrist_xfs':wrist,'landmark_uncertainty_sigmas':np.maximum(np.logaddexp(np.float32(0),raw['landmark_uncertainty_sigmas']),np.float32(1e-5)),'hand_existence':raw['hand_existence'].copy()}
        if 'skel_scales' in raw:result['skel_scales']=np.exp(raw['skel_scales'])
        return result
    def skin(self,angles,wrist_xfs,right_hands=None,axes=None,positions=None):
        """Original defaultProfile 17-frame linear blend skinning in meters.

        right_hands mirrors original left skeleton axes/positions/mesh in X.
        Caller must supply matching *right-handed physical wrist transforms*;
        when using mirrored virtual cameras follow recover_world_wrist below.
        Returns 21 landmarks, all 788 skinned vertices, 22 joint-axis positions.
        """
        n=len(angles);axes=np.broadcast_to(self.w['joint_rotation_axes'] if axes is None else axes,(n,22,3)).copy()
        pos=np.broadcast_to(self.w['joint_rest_positions'] if positions is None else positions,(n,22,3)).copy()
        verts=np.broadcast_to(self.w['mesh_rest_positions'],(n,788,3)).copy()
        mask=np.zeros(n,bool) if right_hands is None else np.asarray(right_hands,bool)
        axes[mask,:,1:]*=-1;pos[mask,:,0]*=-1;verts[mask,:,0]*=-1
        local=joint_local_xfs(axes,pos,np.asarray(angles,np.float32));wrist=np.asarray(wrist_xfs,np.float32)
        frames=[wrist,wrist];joint_pos=np.zeros((n,22,3),np.float32)
        for finger in range(5):
            parent=wrist
            for j in range(4*finger,4*finger+4):
                joint_pos[:,j]=(parent[:,:3,:3]@pos[:,j,:,None])[...,0]+parent[:,:3,3]
                parent=parent@local[:,j]
                if j%4>=1:frames.append(parent)
        joint_pos[:,20:]=wrist[:,None,:3,3]
        frames=np.stack(frames,axis=1);hom=np.concatenate((verts,np.ones((n,len(verts[0]),1),np.float32)),axis=-1)
        skinned=np.einsum('nbij,nvj,vb->nvi',frames,hom,self.w['mesh_skin_weights'],optimize=True)[...,:3]
        sv=self.meta['special_vertices'];tips=skinned[:,sv['fingerTipIndices']]
        lm=np.concatenate((tips,wrist[:,None,:3,3],joint_pos[:,[2,3,4,6,7,8,10,11,12,14,15,16,18,19]],skinned[:,[sv['palmCenterIndex']]]),axis=1)
        return {'landmarks':lm,'mesh_vertices':skinned,'joint_positions':joint_pos,'landmark_order':['thumb_tip','index_tip','middle_tip','ring_tip','pinky_tip','wrist','thumb_mcp','thumb_ip','index_mcp','index_pip','index_dip','middle_mcp','middle_pip','middle_dip','ring_mcp','ring_pip','ring_dip','pinky_mcp','pinky_pip','pinky_dip','palm_center']}


def recover_world_wrist(wrist_in_left_camera0,T_world_left_camera0,right_hands):
    """Match public UmeTrack: inverse cropped extrinsics @ left wrist, then flip
    right wrist first column. T_world_left_camera0 includes crop-camera mirror
    for right hands; the resulting wrist and mirrored skeleton are physical.
    """
    out=np.asarray(T_world_left_camera0,np.float32)@wrist_in_left_camera0
    out[np.asarray(right_hands,bool),:,0]*=-1
    return out
