foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(_arch LESS 70)
    message(FATAL_ERROR "QDK Requires Volta+ for CUDA Builds")
  endif()
endforeach()

find_package(CUDAToolkit REQUIRED)
foreach(cuda_dep cublas cublasLt cusolver)
  if(NOT TARGET CUDA::${cuda_dep})
    message(FATAL_ERROR "Could not find CUDA::${cuda_dep}")
  endif()
endforeach()

include(${CMAKE_CURRENT_LIST_DIR}/cutensor.cmake)
