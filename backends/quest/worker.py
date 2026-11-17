#!/usr/bin/env python3
"""Real shared-camera -> original-weight Hexagon delegates -> skeletal candidates.

No sensor ownership, Mercury, made-up hand poses or image replay. Candidate
validity is an explicit research policy, not recovered Quest native confidence.
"""
import argparse,dataclasses,json,os,threading,time,traceback,signal
from xr_hand.publisher import Publisher
from xr_hand.protocol import StateLog
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from camera import Calibration,FrameReceiver,prepare_crops,model_camera_geometry,rigid_inverse
from pipeline_client import PipelineClient
from decoder import DPEHostStages,model_geometry,prepare_temporal_memory,recover_world_wrist

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--camera-exposure-profile',choices=('kernel-phase2','vr700-fixed-controls-v1'),default='kernel-phase2')
p.add_argument('--assets',type=Path,required=True);p.add_argument('--state-socket',required=True);p.add_argument('--camera-socket',default='/run/quest-hand-camera/input.sock');p.add_argument('--calibration',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=300);p.add_argument('--seconds',type=float,default=60);p.add_argument('--detector-period',type=float,default=1.0);p.add_argument('--detect-threshold',type=float,default=.6);p.add_argument('--exist-threshold',type=float,default=.5);p.add_argument('--save-every',type=int,default=30);p.add_argument('--roi-file',type=Path,help='Observed human-annotated initializer; never represents automatic detection');p.add_argument('--exercise-inactive',action='store_true',help='Exercise both DSP stages on real pixels when no hand is detected; all inactive slots remain invalid');p.add_argument('--compute-only',action='store_true',help='Fixed ROIs on current real camera pixels; always invalid, no tracking claim');a=p.parse_args()
assert 0<=a.frames<=10000 and 0<=a.seconds<=3600 and a.detector_period>0 and a.save_every>=0
assert not a.compute_only or a.roi_file,'Compute-only requires explicit fixed source windows'
a.output.mkdir(parents=True,exist_ok=True);cal=Calibration.load(a.calibration,rig_convention='rig-to-camera');host=DPEHostStages(a.assets);receiver=FrameReceiver(a.camera_socket,a.camera_exposure_profile)
manual_document=json.loads(a.roi_file.read_text()) if a.roi_file else None
manual_rois=manual_document.get('rois') if isinstance(manual_document,dict) else manual_document
if manual_rois is not None:
 if len(manual_rois)!=4 or any(len({x['slot'] for x in manual_rois if x['hand']==h})!=2 for h in [0,1]):raise ValueError('Observed initializer needs two distinct camera ROIs per hand')
latest=None;read_errors=[];lock=threading.Lock();stop=threading.Event()

def capture():
 global latest
 while not stop.is_set():
  try:
   f=receiver.receive(timeout=.5)
   with lock:latest=f
  except (TimeoutError,EOFError):continue
  except Exception as e:
   read_errors.append({'monotonic_ns':time.monotonic_ns(),'error':str(e)});time.sleep(.01)

def safe(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [safe(x) for x in v]
 return v

def rois_from_detection(centers,radii,confidence):
 rois=[];active=np.zeros(2,bool);chosen={}
 for hand in [0,1]:
  choices=[]
  for slot in range(4):
   cx,cy=centers[slot,hand];r=float(radii[slot,hand]);score=float(confidence[slot,hand])
   if np.isfinite([cx,cy,r,score]).all() and score>=a.detect_threshold and 5<=r<=600 and -r<cx<640+r and -r<cy<480+r:
    choices.append((score,slot,[float(cx-r),float(cy-r),float(cx+r),float(cy+r)]))
  choices.sort(reverse=True);active[hand]=len(choices)>=2
  if active[hand]:selection=choices[:2]
  else:
   # Static batch has two hands. Inactive slots still contain real camera
   # pixels, but are never emitted as detected/tracked hands or retained state.
   selection=[(0.,0,[240.,160.,400.,320.]),(0.,1,[240.,160.,400.,320.])]
  chosen[hand]=[{'score':score,'slot':slot,'bbox':box} for score,slot,box in selection]
  rois.extend({'hand':hand,'slot':slot,'bbox':box} for _,slot,box in selection)
 return rois,active,chosen

def recursive_rois(points,frame,previous_detection,active):
 wr=frame.world_from_body@cal.body_from_reference;reference_from_world=rigid_inverse(wr);out=[]
 for hand in [0,1]:
  if not active[hand]:
   for slot in [0,1]:out.append({'hand':hand,'slot':slot,'bbox':[240.,160.,400.,320.]})
   continue
  world=points[hand];ref=world@reference_from_world[:3,:3].T+reference_from_world[:3,3]
  for rec in previous_detection[hand]:out.append({'hand':hand,'slot':rec['slot'],'points_reference':ref.tolist()})
 return out

print(json.dumps({'kind':'listener_ready','camera_socket':a.camera_socket,'output':str(a.output)}),flush=True)
signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
publisher=Publisher(a.state_socket)
t=threading.Thread(target=capture,daemon=True);t.start();start=time.monotonic();last_seq=0;last_detection=-1e9;count=0;memory=np.zeros((2,18,4,4),np.float32);previous_e=None;use_memory=np.zeros(2,bool);landmarks=None;detection=None;active=np.zeros(2,bool);phase_latencies=[];last_exposure=None;last_producer=None;last_world=None;stale_emitted=False
try:
 with PipelineClient(os.environ['QUEST_PIPELINE_SOCKET']) as dsp,StateLog(a.output/'frames.jsonl',publisher) as log,(a.output/'publication.jsonl').open('w') as publication:
  while not stop.is_set() and (not a.frames or count<a.frames) and (not a.seconds or time.monotonic()-start<a.seconds):
   with lock:f=latest
   if f is None or (f.sequence==last_seq and f.producer_pid==last_producer):
    stale=(f is None and time.monotonic()-start>.25) or (f is not None and time.monotonic_ns()-f.received_monotonic_ns>250_000_000)
    if stale and not stale_emitted:
     memory[:]=0;use_memory[:]=False;active[:]=False;previous_e=None;landmarks=None;last_detection=-1e9;stale_emitted=True
     log.write(json.dumps({'kind':'source_stale','sequence':last_seq,'producer_pid':last_producer,'monotonic_ns':time.monotonic_ns(),'hands':[{'hand':h,'valid':False,'reason':'source_stale'} for h in [0,1]]})+'\n');log.flush()
    time.sleep(.002);continue
   stale_emitted=False
   frame_start=time.monotonic_ns();now_boot=time.clock_gettime_ns(time.CLOCK_BOOTTIME);clock_end=time.monotonic_ns()
   exposure=min(f.exposure_start_boottime_ns);exposure_mono=(frame_start+clock_end)//2-(now_boot-exposure)
   epoch_changed=last_producer is not None and (f.producer_pid!=last_producer or f.sequence<last_seq or last_exposure is None or not 0<exposure-last_exposure<250_000_000 or np.linalg.norm(f.world_from_body[:3,3]-last_world[:3,3])>1.0 or np.trace(f.world_from_body[:3,:3]@last_world[:3,:3].T)<1)
   if epoch_changed:
    memory[:]=0;use_memory[:]=False;active[:]=False;previous_e=None;landmarks=None;last_detection=-1e9
   last_seq=f.sequence;last_producer=f.producer_pid;last_exposure=exposure;last_world=f.world_from_body.copy()
   base={'sequence':f.sequence,'producer_pid':f.producer_pid,'history_reset':epoch_changed,'clock_bridge_bracket_ns':clock_end-frame_start,'source_frame_ids':f.source_frame_ids,'exposure_phase':f.exposure_phase,'camera_exposure_profile':f.exposure_profile,'exposure_start_boottime_ns':f.exposure_start_boottime_ns,'exposure_center_boottime_ns':f.exposure_center_boottime_ns,'head_pose_boottime_ns':f.head_pose_boottime_ns,'camera_clock_bracket_ns':f.camera_clock_bracket_ns,'source_received_monotonic_ns':f.received_monotonic_ns,'processing_start_monotonic_ns':frame_start,'stage':'candidate_only','calibration_optically_verified':False,'nimble_equivalence_verified':False,'private_confidence_policy_recovered':False,'physical_wearer_verified':False,'valid_until_boottime_ns':exposure+250_000_000}
   if not 0<=now_boot-exposure<=250_000_000:
    memory[:]=0;use_memory[:]=False;active[:]=False;previous_e=None;landmarks=None;last_detection=-1e9
    base.update(hands=[{'hand':h,'valid':False,'reason':'stale_camera_group'} for h in [0,1]]);log.write(json.dumps(safe(base))+'\n');log.flush();continue
   detected_this_frame=False;detector_start=time.monotonic_ns()
   if manual_rois is not None and (detection is None or a.compute_only):
    rois=manual_rois;active[:]=True;detection={h:[{'slot':x['slot'],'bbox':x['bbox'],'score':None,'source':'human_observed_roi'} for x in manual_rois if x['hand']==h] for h in [0,1]};last_detection=time.monotonic()
   elif manual_rois is None and (time.monotonic()-last_detection>a.detector_period or not active.any()):
    detected_this_frame=True
    pixels=np.ascontiguousarray(np.stack(f.images)[:,None])
    if a.save_every and not (a.output/'first-detector-input.u8').exists():
     (a.output/'first-detector-input.u8').write_bytes(pixels.tobytes());(a.output/'first-detector-input.json').write_text(json.dumps(safe(base),indent=2)+'\n')
    det=dsp.detector(pixels,sequence=f.sequence,exposure_ns=exposure_mono)
    values=np.frombuffer(det.data,'<f4');centers=values[:16].reshape(4,2,2);radii=values[16:24].reshape(4,2);conf=values[24:32].reshape(4,2)
    rois,active,detection=rois_from_detection(centers,radii,conf);last_detection=time.monotonic();base['detector']=dict(det.telemetry(),centers=centers,radii=radii,confidence=conf);use_memory&=active
    if a.save_every and count%a.save_every==0:
     for slot,image in enumerate(f.images):(a.output/f'frame-{count:05d}.slot{slot}.y8').write_bytes(image.tobytes())
   elif landmarks is not None:
    rois=recursive_rois(landmarks,f,detection,active)
    if manual_rois is not None:rois=[r for r in rois if active[r['hand']]]+[r for r in manual_rois if not active[r['hand']]]
   geometry_start=time.monotonic_ns();base['detector_elapsed_ms']=(geometry_start-detector_start)/1e6 if detected_this_frame else 0.;base['detector_age_ms']=(time.monotonic()-last_detection)*1000;base['independent_detection_this_frame']=detected_this_frame
   base['detected_hands']=active.copy() if manual_rois is None else None;base['initializer']='automatic_DETV62' if manual_rois is None else 'human_observed_roi';base['manual_initializer']=manual_document if manual_rois is not None and count==0 else None
   base['inactive_computational_exercise']=bool(a.exercise_inactive and not active.any())
   if not active.any() and not a.exercise_inactive and manual_rois is None:
    use_memory[:]=False;memory[:]=0;base.update(hands=[{'hand':h,'valid':False,'reason':'no_stereo_detection'} for h in [0,1]])
    emitted=time.clock_gettime_ns(time.CLOCK_BOOTTIME);base['exposure_to_status_ms']=(emitted-exposure)/1e6;log.write(json.dumps(safe(base))+'\n');log.flush();count+=1;continue
   try:
    crops,crop_records=prepare_crops(f,cal,rois);K,E=model_camera_geometry(f,cal,crop_records);g=model_geometry(K,E,previous_e,use_memory);mem=prepare_temporal_memory(memory,g);pre_end=time.monotonic_ns()
    back=dsp.backbone(crops,sequence=f.sequence,exposure_ns=exposure_mono);bridge_start=time.monotonic_ns();q,context=host.prepare_fusion(back.data,g['T_camera_image'],g['T_canonical_image']);bridge_end=time.monotonic_ns()
    fused=dsp.fusion(q,sequence=f.sequence,exposure_ns=exposure_mono);head_start=time.monotonic_ns();raw=host.finish(fused.data,context,g['T_camera0_canonical'],(np.ones(2,bool) if manual_rois is not None else active),g['T_curFrame_prevFrame'],mem);decoded=host.decode(raw);wrist=recover_world_wrist(decoded['wrist_xfs'],g['T_world_left_camera0'],[False,True]);skinned=host.skin(decoded['joint_angles'],wrist,[False,True]);landmarks=skinned['landmarks'];head_end=time.monotonic_ns();memory=raw['rnn_mem_features'];previous_e=g['world_to_camera0'];hands=[]
    for hand in [0,1]:
     reasons=['compute_only_no_hand_validation'] if a.compute_only else []
     if not active[hand] and manual_rois is None:reasons.append('no_stereo_detection')
     if manual_rois is not None and not active[hand] and time.monotonic()-last_detection>10:reasons.append('manual_initializer_expired')
     existence=float(raw['hand_existence'][hand]);visibility=raw['encoder_visibility_prediction'][hand*2:hand*2+2].reshape(-1);repro=[]
     if not all(np.isfinite(x).all() for x in [existence,visibility,wrist[hand],landmarks[hand],decoded['joint_angles'][hand],decoded['landmark_uncertainty_sigmas'][hand]]):reasons.append('nonfinite_candidate')
     if existence<a.exist_threshold:reasons.append('low_hand_existence')
     if time.clock_gettime_ns(time.CLOCK_BOOTTIME)-exposure>250_000_000:reasons.append('output_exceeded_source_expiry')
     world_from_ref=f.world_from_body@cal.body_from_reference;ref=landmarks[hand]@rigid_inverse(world_from_ref)[:3,:3].T+rigid_inverse(world_from_ref)[:3,3]
     for rec in detection[hand]:
      cam=cal.cameras[rec['slot']];pt=ref@cam.camera_from_reference[:3,:3].T+cam.camera_from_reference[:3,3];uv=cam.project(pt);box=rec['bbox'];margin=max(box[2]-box[0],box[3]-box[1])*.4
      inside=(pt[:,2]>0)&(uv[:,0]>=box[0]-margin)&(uv[:,0]<=box[2]+margin)&(uv[:,1]>=box[1]-margin)&(uv[:,1]<=box[3]+margin)
      repro.append({'slot':rec['slot'],'inside_expanded_detector_box_fraction':float(inside.mean()),'independent_current_detection':detected_this_frame,'min_depth_m':float(pt[:,2].min()),'image_landmarks':uv})
     if any(x['min_depth_m']<=.03 or x['min_depth_m']>2.5 or (detected_this_frame and x['inside_expanded_detector_box_fraction']<.8) for x in repro):reasons.append('geometry_detection_inconsistent')
     valid=not reasons;use_memory[hand]=valid;active[hand]=valid
     hands.append({'hand':hand,'valid':valid,'validity_scope':('human observed initializer, learned existence and geometry gates' if manual_rois is not None else 'research detector/existence/geometric consistency, not Quest native policy'),'reasons':reasons,'hand_existence':existence,'joint_limit_violations':int(np.sum((decoded['joint_angles'][hand]<host.w['joint_limits'][:,0])|(decoded['joint_angles'][hand]>host.w['joint_limits'][:,1]))),'encoder_visibility':visibility,'wrist_world':wrist[hand],'joint_angles':decoded['joint_angles'][hand],'landmarks_world_m':landmarks[hand],'landmark_order':skinned['landmark_order'],'uncertainty':decoded['landmark_uncertainty_sigmas'][hand],'reprojection':repro})
    emit_boot=time.clock_gettime_ns(time.CLOCK_BOOTTIME);e2e=(emit_boot-exposure)/1e6;phase_latencies.append(e2e)
    base.update(crop_records=crop_records,geometry_canonical_focal=g['canonical_focal_length'],geometry_focal_original_binary_verified=g['canonical_focal_original_binary_verified'],ftl_affine_direction_original_binary_verified=g['ftl_affine_direction_original_binary_verified'],private_nimble_numeric_equivalence=g['private_nimble_numeric_equivalence'],backbone=back.telemetry(),fusion=fused.telemetry(),hands=hands,timing={'geometry_preprocess_ms':(pre_end-geometry_start)/1e6,'detector_ms':base['detector_elapsed_ms'],'feature_bridge_ms':(bridge_end-bridge_start)/1e6,'host_neural_and_decode_ms':(head_end-head_start)/1e6,'exposure_to_decoded_output_ms':e2e,'processing_ms':(head_end-frame_start)/1e6,'decoded_boottime_ns':emit_boot,'clock_scope':'QCFT software exposure-start formula to CLOCK_BOOTTIME decode completion, not optical motion-to-photon'})
    if a.save_every and count%a.save_every==0:
     (a.output/f'frame-{count:05d}.crops.u8').write_bytes(crops.tobytes());(a.output/f'frame-{count:05d}.backbone.u8').write_bytes(back.data);(a.output/f'frame-{count:05d}.fusion.u8').write_bytes(fused.data);np.savez_compressed(a.output/f'frame-{count:05d}.decoded.npz',**raw,landmarks=landmarks,wrist_world=wrist)
   except ValueError as ex:
    active[:]=False;use_memory[:]=False;memory[:]=0;base.update(hands=[{'hand':h,'valid':False,'reason':str(ex)} for h in [0,1]])
   log.write(json.dumps(safe(base))+'\n');log.flush();
   if publication.tell()>2*1024*1024:publication.seek(0);publication.truncate()
   published=time.clock_gettime_ns(time.CLOCK_BOOTTIME);publication.write(json.dumps({'sequence':f.sequence,'producer_pid':f.producer_pid,'record_flushed_boottime_ns':published,'exposure_to_file_publication_ms':(published-exposure)/1e6,'boundary':'JSONL write+flush complete; no display or consumer receipt measured'})+'\n');publication.flush();print(json.dumps({'kind':'hand_candidate','frame':count,'sequence':f.sequence,'valid':[x.get('valid',False) for x in base['hands']],'exposure_to_output_ms':base.get('timing',{}).get('exposure_to_decoded_output_ms')}),flush=True);count+=1
finally:
 stop.set();t.join(timeout=2);receiver.close();publisher.close();(a.output/'reader-errors.json').write_text(json.dumps(read_errors,indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps({'frames':count,'decoded_frames':len(phase_latencies),'device_executed':bool(count),'full_two_hand_tracking_accepted':False,'optical_latency_calibrated':False,'numpy_host_head_only':True,'dsp_backbone_fusion':True,'inactive_computational_exercise_enabled':a.exercise_inactive,'fixed_roi_compute_only':a.compute_only,'elapsed_seconds':time.monotonic()-start,'error_count':len(read_errors),'latency_ms':{str(q):float(np.percentile(phase_latencies,q)) for q in [50,95,99]} if phase_latencies else None},indent=2)+'\n')
if not count:raise RuntimeError('No live frame processed')
