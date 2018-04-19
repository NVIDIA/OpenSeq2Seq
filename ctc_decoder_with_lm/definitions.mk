# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2018 Mozilla Corporation
#
# The code was taken from Mozilla DeepSpeech project:
# https://github.com/mozilla/DeepSpeech/tree/master/native_client

NC_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

TARGET    ?= host
TFDIR     ?= $(abspath $(NC_DIR)/../../tensorflow)
PREFIX    ?= /usr/local
SO_SEARCH ?= $(TFDIR)/bazel-bin/

ifeq ($(TARGET),host)
TOOLCHAIN       :=
CFLAGS          :=
LDFLAGS         :=
SOX_CFLAGS      := `pkg-config --cflags sox`
SOX_LDFLAGS     := `pkg-config --libs sox`
PYTHON_PACKAGES := numpy
ifeq ($(OS),Linux)
PYTHON_PLATFORM_NAME := --plat-name manylinux1_x86_64
endif
endif

OS := $(shell uname -s)
CFLAGS  += $(EXTRA_CFLAGS)
LIBS    := -ltensorflow_cc -ltensorflow_framework $(EXTRA_LIBS)
LDFLAGS += -Wl,-rpath,. -L${TFDIR}/bazel-bin/tensorflow -L${TFDIR}/bazel-bin/ctc_decoder_with_lm -L/usr/local/lib/python2.7/dist-packages/tensorflow $(EXTRA_LDFLAGS) $(LIBS)

AS      := $(TOOLCHAIN)as
CC      := $(TOOLCHAIN)gcc
CXX     := $(TOOLCHAIN)c++
LD      := $(TOOLCHAIN)ld
LDD     := $(TOOLCHAIN)ldd $(TOOLCHAIN_LDD_OPTS)

RPATH_PYTHON         := '-Wl,-rpath,\$$ORIGIN/lib/'
RPATH_NODEJS         := '-Wl,-rpath,$$\$$ORIGIN/../'
META_LD_LIBRARY_PATH := LD_LIBRARY_PATH

# Takes care of looking into bindings built (SRC_FILE, can contain a wildcard)
# for missing dependencies and copying those dependencies into the
# TARGET_LIB_DIR. If supplied, MANIFEST_IN will be echo'ed to a list of
# 'include x.so'.
#

define copy_missing_libs
    SRC_FILE=$(1); \
    TARGET_LIB_DIR=$(2); \
    MANIFEST_IN=$(3); \
    echo "Analyzing $$SRC_FILE copying missing libs to $$SRC_FILE"; \
    echo "Maybe outputting to $$MANIFEST_IN"; \
    \
    (mkdir $$TARGET_LIB_DIR || true); \
    missing_libs=""; \
    for lib in $$SRC_FILE; do \
        missing_libs="$$missing_libs $$($(LDD) $$lib | grep 'not found' | awk '{ print $$1 }')"; \
    done; \
    \
    for missing in $$missing_libs; do \
        find $(SO_SEARCH) -type f -name "$$missing" -exec cp {} $$TARGET_LIB_DIR \; ; \
        if [ ! -z "$$MANIFEST_IN" ]; then \
            echo "include $$TARGET_LIB_DIR/$$missing" >> $$MANIFEST_IN; \
        fi; \
    done; \
endef
