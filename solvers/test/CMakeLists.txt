cmake_minimum_required (VERSION 2.6)

add_executable(testConstraint testConstraint.cpp)
include_directories( .. )
set_target_properties(testConstraint PROPERTIES COMPILE_FLAGS "-fPIC")
target_link_libraries(testConstraint drakeConstraint)
