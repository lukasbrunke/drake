# -*- python -*-
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
#def conex_repository(
#        name,
#        mirrors = None):
#    native.local_repository(
#        name = name,
#        path = "/home/frank/conexnew/conex/",
#    )


def conex_repository(
        name,
        mirrors = None):
      git_repository(
          name = name,
          commit = "7a589c3f9279703d4541a359e38ee722e23fce5d",
          #shallow_since = "1611401753 -0500",
          remote = "git@github.com:frankpermenter/conex.git",
      )
