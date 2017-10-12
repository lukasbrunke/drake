load(
    "//tools:drake.bzl",
    "drake_cc_library",
    "drake_cc_googletest",
)
load("//tools/lint:lint.bzl", "add_lint_tests")

drake_cc_library(
    name = "multicontact_time_optimal_planner",
    srcs = ["multicontact_time_optimal_planner.cc"],
    hdrs = ["multicontact_time_optimal_planner.h"],
    deps = [
        "//drake/solvers:mathematical_program",
    ],
)

drake_cc_googletest(
    name = "multicontact_time_optimal_planner_test",
    deps = [
        ":multicontact_time_optimal_planner",
    ],
)

add_lint_tests()
