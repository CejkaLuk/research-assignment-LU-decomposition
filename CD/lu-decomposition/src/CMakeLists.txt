install( DIRECTORY Decomposition/ DESTINATION ${DECOMPOSITION_TARGET_INCLUDE_DIRECTORY}
         MESSAGE_NEVER
         FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" )

if( ${BUILD_TESTS} )
   add_subdirectory( UnitTests )
endif()

if( ${BUILD_BENCHMARK} )
   add_subdirectory( Benchmark )
endif()