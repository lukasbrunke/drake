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
          commit = "0b8767ca54ebadf748b1ea7795dc87319505adeb",
          #shallow_since = "1611401753 -0500",
          remote = "git@github.com:frankpermenter/conex.git",
      )
