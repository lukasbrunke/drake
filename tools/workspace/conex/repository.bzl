# -*- python -*-
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
#def conex_repository(
#        name,
#        mirrors = None):
#    native.local_repository(
#        name = name,
#        path = "",
#    )
def conex_repository(
        name,
        mirrors = None):
      git_repository(
          name = name,
          commit ="8f495de227dcc8f9c93b4ec9fc9fc2bc5b283775",
          shallow_since = "1611401753 -0500",
          remote = "git@github.com:frankpermenter/conex.git",
      )
