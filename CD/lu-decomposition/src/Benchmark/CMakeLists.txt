#add_subdirectory( Decomp )

if( BUILD_CUDA )
   cuda_include_directories( ${CXX_BENCHMARKS_INCLUDE_DIRS} )
   cuda_add_executable( decomposition-benchmark decomposition-benchmark.cu
                        OPTIONS ${CXX_BENCHMARKS_FLAGS} )
   target_link_libraries( decomposition-benchmark ${CUDA_cusparse_LIBRARY} ${CUDA_cudadevrt_LIBRARY} )
else()
   target_compile_options( decomposition-benchmark  PRIVATE ${CXX_BENCHMARKS_FLAGS} )
   target_include_directories( decomposition-benchmark PRIVATE ${CXX_BENCHMARKS_INCLUDE_DIRS} )
endif()

install( TARGETS decomposition-benchmark RUNTIME DESTINATION bin )