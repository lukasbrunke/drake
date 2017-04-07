# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:cpplint.bzl", "cpplint")
load(
    "//tools:drake.bzl",
    "drake_cc_library",
    "drake_cc_googletest",
)

package(default_visibility = ["//visibility:public"])

drake_cc_library(
    name = "IRB140_analytical_kinematics",
    srcs = ["IRB140_analytical_kinematics.cc"],
    hdrs = ["IRB140_analytical_kinematics.h"],
    deps = [
        "//drake/common",
        "//drake/common:symbolic",
        "//drake/multibody:rigid_body_tree",
        "//drake/multibody:rigid_body_tree_construction",
        "//drake/multibody/parsers",
    ],
)

drake_cc_googletest(
    name = "IRB140_analytical_kinematics_test",
    srcs = ["test/IRB140_analytical_kinematics_test.cc"],
    deps = [
        ":IRB140_analytical_kinematics",
        "//drake/common:eigen_matrix_compare",
    ],
)

cpplint()
