set(drakeForceClosureSources
        force_closure_util.cc)

set(drakeForceClosureHeaders
        force_closure_util.h)

add_library_with_exports(LIB_NAME drakeForceClosure
        SOURCE_FILES ${drakeForceClosureSources} ${drakeForceClosureHeaders})
target_link_libraries(drakeForceClosure
        Eigen3::Eigen)

add_subdirectory(test)