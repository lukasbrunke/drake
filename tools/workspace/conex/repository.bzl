# -*- python -*-
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
def conex_repository(
        name,
        mirrors = None):
    native.local_repository(
        name = name,
        path = "/home/frank/conexnew/conex/",
    )
#def conex_repository(
#        name,
#        mirrors = None):
#      git_repository(
#          name = name,
#          commit = "7260fb3691aeb8ea0151dc80a8e3cd95f3df55d3",
#          #shallow_since = "1611401753 -0500",
#          remote = "git@github.com:frankpermenter/conex.git",
#      )
