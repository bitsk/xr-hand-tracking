# SPDX-License-Identifier: MIT
"""Session-owned original-weight Hexagon backend; never opens a camera."""
import argparse,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,threading
from pathlib import Path

def verified_assets(root):
    m=json.loads((root/'assets.lock.json').read_text())
    if m.get('schema')!=1:raise ValueError('Invalid asset lock')
    required={'backbone-hvx.qhf','fusion-hvx.qhf','detector-hvx.qhf','original-decoder-weights.npz','contract.json','dsp/libhexagon_skel.so'}
    if not required<=set(m['files']):raise ValueError('Missing model asset entries')
    for name,item in m['files'].items():
        p=root/name
        if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink():raise ValueError('Unsafe asset path')
        if p.stat().st_size!=item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Asset checksum mismatch: '+name)
    return m

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.config.read_text())
    if c.get('schema')!=1 or c.get('backend')!='quest-dsp':raise ValueError('Unsupported hand configuration')
    if os.geteuid()!=0 or os.uname().machine!='aarch64':raise RuntimeError('Backend requires native arm64 FastRPC privileges')
    assets=Path(c['assets']);verified_assets(assets)
    if not Path(c['calibration']).is_file():raise FileNotFoundError('Private camera calibration must be configured')
    run=Path(c['camera_socket']).parent.parent;run.mkdir(parents=True,exist_ok=True)
    lock=(run/'backend.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logdir=Path(c['log_directory']);logdir.mkdir(parents=True,exist_ok=True)
    stopped=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stopped.set())
    signal.signal(signal.SIGINT,lambda *_:stopped.set())
    for attempt in range(2):
        try:
            run_backend(c,assets,run,logdir,stopped)
            return
        except DspExecution1042 as error:
            if stopped.is_set():return
            if attempt:raise
            print(json.dumps({'kind':'dsp_recovery','code':1042,'attempt':1,
                              'private_pd_cleaned':True,'detail':str(error)}),flush=True)
            # The failed consumer is gone and no stale gesture survives. Only
            # one fresh private PD is allowed; head/camera owners stay running.
            if stopped.wait(.5):return

class DspExecution1042(RuntimeError):
    pass

def run_backend(c,assets,run,logdir,stopped):
    runtime=run/'qcom'
    if runtime.exists():shutil.rmtree(runtime)
    shutil.copytree(c['runtime_root'],runtime,symlinks=False)
    for file in (assets/'dsp').iterdir():shutil.copy2(file,runtime/'dsp/cdsp'/file.name)
    stage=run/'stages.sock';stage.unlink(missing_ok=True)
    for path in [Path(c['camera_socket']),Path(c['state_socket'])]:
        if path.exists():raise FileExistsError('Endpoint already owned: '+str(path))
    env=dict(os.environ,QUEST_HAND_BUNDLE=str(assets),QUEST_HAND_RESULT=str(logdir),QUEST_PIPELINE_RESULT=str(logdir),XR_FASTRPC_UNSIGNED_PD='1',QUEST_PIPELINE_SOCKET=str(stage),QUEST_PIPELINE_IDLE_MS='60000',QUEST_PIPELINE_DETECTOR='1',QUEST_PIPELINE_THREADS=str(c.get('threads',1)),QUEST_PIPELINE_VTCM_BYTES=str(c.get('vtcm_bytes',16384)),QUEST_PIPELINE_DEBUG_LEVEL='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    env['PYTHONPATH']='/usr/share/xr-hand-tracking/python'+os.pathsep+str(Path(__file__).resolve().parents[2]/'backends/quest')
    dsp=None;worker=None;log_thread=None;ready=threading.Event()
    fault1042=[];clean_exit=threading.Event();module_closed=threading.Event()
    def dsp_output():
        for line in dsp.stdout:
            print(line,end='',flush=True)
            if 'stage=pipeline_ready ' in line:ready.set()
            if ('stage=graph_execute_error ' in line or 'stage=detector_execute_error ' in line) and 'transport=0 code=1042 ' in line:
                fault1042.append(line.strip())
            if 'stage=module_close rc=0 loader_error=0' in line:module_closed.set()
            if 'stage=plugin_exit ' in line and ' live=0 errors=0' in line:clean_exit.set()
    try:
        # DSP diagnostics go to the bounded system journal, including cleanup.
        dsp=subprocess.Popen([c['rpc_daemon'],'-f','/dev/fastrpc-cdsp','-c',str(runtime/'dsp/cdsp/fastrpc_shell_unsigned_3'),'-d','cdsp','-R',str(runtime),'-P',c['dsp_client']],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors="replace")
        log_thread=threading.Thread(target=dsp_output,daemon=True);log_thread.start()
        deadline=time.monotonic()+50
        while not stopped.is_set() and dsp.poll() is None and time.monotonic()<deadline:
            if stage.exists() and ready.is_set():break
            time.sleep(.05)
        else:
            if stopped.is_set():return
            raise RuntimeError('DSP did not become ready; see the system journal')
        worker_path=Path(__file__).resolve().parents[2]/'backends/quest/worker.py'
        command=[sys.executable,str(worker_path),'--assets',str(assets),'--camera-socket',c['camera_socket'],'--state-socket',c['state_socket'],'--calibration',c['calibration'],'--output',str(logdir),'--frames','0','--seconds','0','--save-every','0','--detector-period',str(c.get('detector_period',1.))]
        if c.get('exercise_inactive',False):command.append('--exercise-inactive')
        worker=subprocess.Popen(command,env=env)
        print(json.dumps({'kind':'hand_service_ready','backend':'quest-dsp','worker_pid':worker.pid,'state_socket':c['state_socket']}),flush=True)
        while not stopped.is_set():
            if dsp.poll() is not None or worker.poll() is not None:
                if dsp.poll() is None:
                    try:dsp.wait(timeout=2)
                    except subprocess.TimeoutExpired:pass
                if log_thread:log_thread.join(timeout=1)
                if dsp.poll() is not None and fault1042 and clean_exit.is_set() and module_closed.is_set():
                    raise DspExecution1042(fault1042[-1])
                raise RuntimeError('Hand backend exited; see the system journal')
            time.sleep(.1)
    finally:
        if worker and worker.poll() is None:
            worker.terminate()
            try:worker.wait(timeout=5)
            except subprocess.TimeoutExpired:worker.kill();worker.wait()
        if dsp and dsp.poll() is None:
            sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'backends/quest'))
            try:
                from pipeline_client import PipelineClient
                with PipelineClient(stage,timeout=3) as client:client.shutdown()
            except (OSError,RuntimeError):pass
            try:dsp.wait(timeout=4)
            except subprocess.TimeoutExpired:dsp.terminate();dsp.wait(timeout=3)
        if log_thread:log_thread.join(timeout=1)
        stage.unlink(missing_ok=True)
        shutil.rmtree(runtime,ignore_errors=True)
