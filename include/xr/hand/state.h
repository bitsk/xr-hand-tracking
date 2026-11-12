// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>
/* Little endian, IEEE754 float32, SOCK_SEQPACKET. No file descriptors.
 * CLOCK_BOOTTIME timestamps; positions in metres in the raw OpenVR driver
 * world used by the head tracker (before runtime origin/recenter transforms).
 * Pose layout is position xyz, quaternion xyzw. Landmark order is specified
 * in README. Source must be LIVE for production input; replay is never live.
 */
#define XR_HAND_MAGIC UINT32_C(0x48445258)
#define XR_HAND_VERSION 1
#define XR_HAND_SOURCE_LIVE 1
#define XR_HAND_SOURCE_REPLAY 2
#define XR_HAND_VALID 1
#define XR_HAND_PINCH 2
#define XR_HAND_GRAB 4
struct xr_hand_state {
    uint32_t hand; /* 0 left, 1 right (model slots, not screen sides) */
    uint32_t flags;
    float grip[7], aim[7];
    float pinch, grab;
    float landmarks[21][3];
};
struct xr_hand_packet {
    uint32_t magic, version, bytes, source;
    uint64_t sequence, exposure_boottime_ns, decoded_boottime_ns, valid_until_boottime_ns;
    uint32_t producer_pid, reserved;
    struct xr_hand_state hands[2];
};
#if defined(__cplusplus)
static_assert(sizeof(xr_hand_state) == 324, "hand ABI");
static_assert(sizeof(xr_hand_packet) == 704, "packet ABI");
#else
_Static_assert(sizeof(struct xr_hand_packet) == 704, "packet ABI");
#endif
