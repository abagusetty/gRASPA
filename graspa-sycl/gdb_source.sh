#!/usr/bin/env bash

export ZE_AFFINITY_MASK=0.0
export IGC_VISAOptions="-enableBarrierWA"
export NEO_CACHE_PERSISTENT=0
export SYCL_CACHE_PERSISTENT=0
export DisableScratchPages=0
export ZET_ENABLE_PROGRAM_DEBUGGING=1
export SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR=1
export UR_L0_SERIALIZE=2
