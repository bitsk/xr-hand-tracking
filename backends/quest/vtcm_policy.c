// SPDX-License-Identifier: MIT
// Replacement for the legacy allocator's choose-size entry, in a private copy
// of the one hash-verified library. The graph workspace field is at byte 0x90.
// Avoid using a transient free-space snapshot as the required allocation size.
// Keep the original acquire/release and operator implementations unchanged.
int xr_quest_vtcm_choose_size(void *graph)
{
    *(unsigned int *)((unsigned char *)graph + 0x90) = 65536;
    return 0;
}
