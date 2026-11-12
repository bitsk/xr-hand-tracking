#!/usr/bin/env python3
"""Client for two persistent original-weight Hexagon stages.

The caller supplies the real geometry/requantization bridge between stages.
This module intentionally never connects tensors by a guessed reshape.
No camera producer or learned CPU fallback is introduced.
"""
from dataclasses import asdict,dataclass
import argparse
import json
import math
from pathlib import Path
import socket
import struct
import time

REQUEST=struct.Struct('<4sIQQII')
RESPONSE=struct.Struct('<4sIQQQQQQiI')
STAGES={
    'backbone':{'op':1,'input_shape':(4,96,96,1),'output_shape':(4,4,4,60),'input_bytes':36864,'output_bytes':3840,'input_scale':1/255,'input_zero':0,'output_scale':0.008666601032018661,'output_zero':143},
    'detector':{'op':3,'input_shape':(4,1,480,640),'output_shapes':((4,2,2),(4,2),(4,2)),'input_bytes':1228800,'output_bytes':128,'input_order':'NCHW','output_dtype':'float32'},
    'fusion':{'op':2,'input_shape':(2,4,4,120),'output_shape':(2,4,4,60),'input_bytes':3840,'output_bytes':1920,'input_scale':0.006989060435444117,'input_zero':143,'output_scale':0.006207913625985384,'output_zero':110},
}

@dataclass(frozen=True)
class TensorResult:
    stage:str
    sequence:int
    exposure_ns:int
    server_header_received_ns:int
    rpc_start_ns:int
    rpc_end_ns:int
    server_response_ready_ns:int
    client_send_ns:int
    client_received_ns:int
    quantized_min:float|None
    quantized_max:float|None
    data:bytes

    def telemetry(self):
        result=asdict(self);result.pop('data')
        result['data_bytes']=len(self.data)
        result['rpc_ms']=(self.rpc_end_ns-self.rpc_start_ns)/1e6
        result['client_roundtrip_ms']=(self.client_received_ns-self.client_send_ns)/1e6
        result['exposure_to_client_output_ms']=(self.client_received_ns-self.exposure_ns)/1e6 if self.exposure_ns else None
        return result

class PipelineClient:
    """One target-side socket session; accepts bytes, arrays or mmap views."""
    def __init__(self,path,timeout=30):
        self.socket=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        try:self.socket.connect(str(path))
        except BaseException:self.socket.close();raise
        self.closed=False

    def close(self):
        if not self.closed:self.socket.close();self.closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def _read(self,count):
        data=bytearray(count);view=memoryview(data);offset=0
        while offset<count:
            n=self.socket.recv_into(view[offset:])
            if not n:raise EOFError('DSP stage server closed the connection')
            offset+=n
        return bytes(data)

    def _call(self,op,data,sequence,exposure_ns):
        view=memoryview(data).cast('B')
        send_ns=time.monotonic_ns()
        self.socket.sendall(REQUEST.pack(b'QDP1',op,sequence,exposure_ns,len(view),0))
        if view:self.socket.sendall(view)
        response=RESPONSE.unpack(self._read(RESPONSE.size))
        magic,returned_op,returned_sequence,returned_exposure,received,start,end,ready,status,count=response
        if (magic,returned_op,returned_sequence,returned_exposure)!=(b'QDR1',op,sequence,exposure_ns):raise RuntimeError('Stage response identity mismatch')
        if status or count>3848:raise RuntimeError(f'Stage response error: status={status}, bytes={count}')
        payload=self._read(count);client_received=time.monotonic_ns()
        if not op:
            if payload:raise RuntimeError('Shutdown returned an unexpected tensor')
            return None
        name={1:'backbone',2:'fusion',3:'detector'}[op]
        if op==3:
            if len(payload)!=128:raise RuntimeError('Detector tensor length mismatch')
            if not all(math.isfinite(x) for x in struct.unpack('<32f',payload)):raise RuntimeError('Nonfinite detector tensor')
            if not (received<=start<=end<=ready):raise RuntimeError('Stage timestamp order')
            return TensorResult(name,sequence,exposure_ns,received,start,end,ready,send_ns,client_received,None,None,payload)
        if len(payload)!=STAGES[name]['output_bytes']+8:raise RuntimeError('Stage tensor length mismatch')
        lo,hi=struct.unpack_from('<ff',payload)
        if not all(map(math.isfinite,(lo,hi))) or hi<=lo:raise RuntimeError('Invalid stage quantization range')
        if not (received<=start<=end<=ready):raise RuntimeError('Stage timestamp order')
        return TensorResult(name,sequence,exposure_ns,received,start,end,ready,send_ns,client_received,lo,hi,payload[8:])

    def execute(self,stage,data,*,sequence,exposure_ns=0):
        spec=STAGES[stage]
        if memoryview(data).nbytes!=spec['input_bytes']:raise ValueError(f'{stage} requires {spec["input_bytes"]} uint8 bytes')
        return self._call(spec['op'],data,sequence,exposure_ns)

    def backbone(self,data,*,sequence,exposure_ns=0):return self.execute('backbone',data,sequence=sequence,exposure_ns=exposure_ns)
    def fusion(self,data,*,sequence,exposure_ns=0):return self.execute('fusion',data,sequence=sequence,exposure_ns=exposure_ns)
    def detector(self,data,*,sequence,exposure_ns=0):return self.execute('detector',data,sequence=sequence,exposure_ns=exposure_ns)
    def shutdown(self):
        self._call(0,b'',0,0);self.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket',required=True)
    p.add_argument('--backbone-input',type=Path)
    p.add_argument('--detector-input',type=Path,help='4x1x480x640 NCHW uint8; server must opt in')
    p.add_argument('--fusion-input',type=Path,help='Explicit independent stage input; not inferred from backbone output')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--shutdown',action='store_true')
    a=p.parse_args();assert 0<a.repeat<=10000
    a.output.mkdir(parents=True,exist_ok=False)
    tensors={name:path.read_bytes() for name,path in [('backbone',a.backbone_input),('fusion',a.fusion_input),('detector',a.detector_input)] if path}
    if not tensors:raise ValueError('At least one explicit stage input is required')
    with PipelineClient(a.socket) as client, (a.output/'telemetry.jsonl').open('w') as stream:
        for seq in range(a.repeat):
            for stage,data in tensors.items():
                result=client.execute(stage,data,sequence=seq)
                extension='f32' if stage=='detector' else 'u8'
                (a.output/f'{seq:06d}-{stage}.{extension}').write_bytes(result.data)
                stream.write(json.dumps(result.telemetry())+'\n');stream.flush()
        if a.shutdown:client.shutdown()

if __name__=='__main__':main()
