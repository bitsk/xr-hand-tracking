# SPDX-License-Identifier: MIT
"""Apply the compiled resource policy only to the private runtime library copy."""
import hashlib
import struct
from pathlib import Path

ORIGINAL_SHA256 = '30cffd7fdeb908139f6756f68887a655942452b0a709aadf946a202c1064a30c'
ENTRY = 0x13ba68
FUNCTION_SIZE = 344


def prepare_runtime_library(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Resource policy requires a separate runtime copy')
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('Unsupported DSP library for VTCM policy')
    payload = Path(__file__).with_name('vtcm-policy.bin').read_bytes()
    if not 0 < len(payload) <= FUNCTION_SIZE or len(payload) % 4:
        raise ValueError('Invalid compiled VTCM policy size')
    offset = struct.unpack_from('<I', original, 28)[0]
    size, count = struct.unpack_from('<HH', original, 42)
    matches = []
    for index in range(count):
        row = struct.unpack_from('<8I', original, offset + index * size)
        kind, file_offset, address, _, file_size, _, flags, _ = row
        if kind == 1 and flags & 1 and address <= ENTRY and ENTRY + FUNCTION_SIZE <= address + file_size:
            matches.append(file_offset + ENTRY - address)
    if len(matches) != 1:
        raise ValueError('VTCM policy function must map to one executable LOAD')
    start = matches[0]
    patched = original[:start] + payload + original[start + len(payload):]
    destination.write_bytes(patched)
    return {'kind': 'inner_vtcm_policy', 'workspace_bytes': 65536,
            'original_sha256': ORIGINAL_SHA256,
            'runtime_sha256': hashlib.sha256(patched).hexdigest(),
            'policy_sha256': hashlib.sha256(payload).hexdigest(),
            'replacement_bytes': len(payload), 'original_assets_unchanged': True}
