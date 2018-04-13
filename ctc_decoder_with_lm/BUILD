# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2018 Mozilla Corporation
#
# The code was taken from Mozilla DeepSpeech project:
# https://github.com/mozilla/DeepSpeech/tree/master/native_client

load("//tensorflow:tensorflow.bzl", "if_linux_x86_64", "tf_cc_shared_object")


tf_cc_shared_object(
    name = "libctc_decoder_with_kenlm.so",
    srcs = [
            "beam_search.cc",
            "beam_search.h",
            "alphabet.h",
            "trie_node.h"
           ] +
           glob(["kenlm/lm/*.cc", "kenlm/util/*.cc", "kenlm/util/double-conversion/*.cc",
                 "kenlm/lm/*.hh", "kenlm/util/*.hh", "kenlm/util/double-conversion/*.h"],
                exclude = ["kenlm/*/*test.cc", "kenlm/*/*main.cc"]) 
          # +  glob(["boost_locale/**/*.hpp"])
    ,
    #includes = ["kenlm", "boost_locale"],
    includes = ["kenlm"],
    copts = ["-std=c++11"],
    defines = ["KENLM_MAX_ORDER=6"],
    deps = ["//tensorflow/core:framework_headers_lib",
            "//tensorflow/core/util/ctc",
            "//third_party/eigen3",
    ],
)

cc_binary(
    name = "generate_trie",
    srcs = [
            "generate_trie.cpp",
            "trie_node.h",
            "alphabet.h",
           ] +
           glob(["kenlm/lm/*.cc", "kenlm/util/*.cc", "kenlm/util/double-conversion/*.cc",
                 "kenlm/lm/*.hh", "kenlm/util/*.hh", "kenlm/util/double-conversion/*.h"],
                exclude = ["kenlm/*/*test.cc", "kenlm/*/*main.cc"]) +
           glob(["boost_locale/**/*.hpp"]),
    includes = ["kenlm", "boost_locale"],
    copts = ["-std=c++11"],
    linkopts = ["-lm"],
    defines = ["KENLM_MAX_ORDER=6"],
)
