# SPDX-License-Identifier: MIT
"""Single-owned hand wire ABI v1. Coordinates match the head driver world."""
import json, os, struct, time
from pathlib import Path
HEADER=struct.Struct('<IIIIQQQQII')
HAND=struct.Struct('<II79f')
MAGIC=0x48445258
SIZE=HEADER.size+2*HAND.size
assert SIZE==704

def decode(packet):
    if len(packet)!=SIZE:raise ValueError('Invalid hand packet length')
    h=HEADER.unpack_from(packet)
    if h[:3]!=(MAGIC,1,SIZE) or h[3] not in (1,2) or h[9]:raise ValueError('Invalid hand ABI')
    hands=[]
    for i in range(2):
        x=HAND.unpack_from(packet,HEADER.size+i*HAND.size)
        if x[0]!=i or x[1]&~7:raise ValueError('Invalid hand flags/order')
        hands.append(dict(hand=['left','right'][i],flags=x[1],valid=bool(x[1]&1),grip=x[2:9],aim=x[9:16],pinch=x[16],grab=x[17],landmarks=[x[18+3*j:21+3*j] for j in range(21)]))
    return dict(sequence=h[4],source='live' if h[3]==1 else 'replay',exposure_boottime_ns=h[5],decoded_boottime_ns=h[6],valid_until_boottime_ns=h[7],producer_pid=h[8],hands=hands)

class StateLog:
    """Bounded diagnostic log plus publication at every inference boundary."""
    def __init__(self,path,publisher):
        self.path=Path(path);self.publisher=publisher;self.file=None
    def __enter__(self):
        self.file=self.path.open('w');return self
    def __exit__(self,*args):self.file.close()
    def write(self,line):
        self.publisher.publish(json.loads(line))
        if self.file.tell()>8*1024*1024:
            self.file.close();self.path.replace(self.path.with_suffix('.jsonl.1'));self.file=self.path.open('w')
        return self.file.write(line)
    def flush(self):self.file.flush()
