cmake_minimum_required(VERSION 3.16)
project(xr_hand_backend_quest_dsp VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
add_library(xr-quest-hand-client SHARED dsp_client.cpp)
option(XR_QUEST_DIAGNOSTIC_INNER_LOG "Use the separately verified private-PD legacy log bridge" OFF)
if(XR_QUEST_DIAGNOSTIC_INNER_LOG)
  target_compile_definitions(xr-quest-hand-client PRIVATE XR_QUEST_DIAGNOSTIC_INNER_LOG=1)
endif()
target_compile_features(xr-quest-hand-client PRIVATE cxx_std_17)
target_compile_options(xr-quest-hand-client PRIVATE -O2 -Wall -Wextra)
install(TARGETS xr-quest-hand-client LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/xr/hand-tracking)
install(FILES camera.py decoder.py pipeline_client.py worker.py contract.json DESTINATION ${CMAKE_INSTALL_DATADIR}/xr-hand-tracking/backends/quest)

# The ARM64 host compiler cannot compile Hexagon code. Build this freestanding
# policy with an explicit host LLVM toolchain, without any proprietary inputs.
find_program(XR_HEXAGON_CLANG NAMES clang-21 REQUIRED)
find_program(XR_HEXAGON_OBJCOPY NAMES llvm-objcopy-21 REQUIRED)
set(VTCM_POLICY ${CMAKE_CURRENT_BINARY_DIR}/vtcm-policy.bin)
add_custom_command(OUTPUT ${VTCM_POLICY}
  COMMAND ${XR_HEXAGON_CLANG} --target=hexagon -mcpu=hexagonv66 -O2
    -ffreestanding -fno-builtin -fPIC -nostdlib -fuse-ld=lld
    -Wl,-Ttext=0x13ba68,--entry=xr_quest_vtcm_choose_size,--build-id=none
    ${CMAKE_CURRENT_SOURCE_DIR}/vtcm_policy.c -o vtcm-policy.elf
  COMMAND ${XR_HEXAGON_OBJCOPY} -O binary --only-section=.text vtcm-policy.elf ${VTCM_POLICY}
  DEPENDS vtcm_policy.c VERBATIM)
add_custom_target(vtcm-policy ALL DEPENDS ${VTCM_POLICY})
install(FILES ${VTCM_POLICY} vtcm_policy.py DESTINATION ${CMAKE_INSTALL_DATADIR}/xr-hand-tracking/backends/quest)
