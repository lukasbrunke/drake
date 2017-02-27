set(drakeForceClosureSources
    force_closure_grasp_optimization.cc
    force_closure_util.cc
    contact_facet.cc)

set(drakeForceClosureHeaders
    force_closure_grasp_optimization.h
    force_closure_util.h
    contact_facet.h)

add_library_with_exports(LIB_NAME drakeForceClosure
        SOURCE_FILES ${drakeForceClosureSources} ${drakeForceClosureHeaders})
target_link_libraries(drakeForceClosure
        drakeCommon
        drakeOptimization
        Eigen3::Eigen
        Qhull::qhullcpp
        Qhull::qhull)

add_subdirectory(test)