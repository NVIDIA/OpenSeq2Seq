FROM tensorflow/tensorflow:1.12.0-devel-gpu-py3

RUN cd /tensorflow; 
RUN ln -s /tensorflow /opt/tensorflow
RUN ls -al /opt/tensorflow 

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/cuda/lib64/stubs:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

ARG PYVER=3.5

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        libibverbs-dev \
        libboost-all-dev \
        sox libsox-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/tensorflow


ENV CUDA_TOOLKIT_PATH /usr/local/cuda
ENV TF_CUDA_VERSION "10.0"
ENV TF_CUDNN_VERSION "7"
ENV CUDNN_INSTALL_PATH /usr/lib/x86_64-linux-gnu
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES "6.0,6.1,7.0"
ENV TF_NEED_GCP 0
ENV TF_NEED_HDFS 0
ENV TF_ENABLE_XLA 1
ENV TF_NEED_TENSORRT 1
ENV CC_OPT_FLAGS "-march=sandybridge -mtune=broadwell"


# Build and install TF

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2

# TensorBoard
EXPOSE 6006

#===KenLM ====================================================
#
WORKDIR /opt

RUN git clone https://github.com/kpu/kenlm
RUN mkdir -p /opt/kenlm/build
RUN cd /opt/kenlm/build && cmake .. && make -j20

#===============OpenSeq2Seq============================================================================

WORKDIR /opt

RUN pip install --upgrade numpy librosa scipy

#------------------------------------------------------------------------------
RUN mkdir -p /opt/OpenSeq2Seq
COPY . /opt/OpenSeq2Seq
RUN cd OpenSeq2Seq/ && pip install -r requirements.txt

RUN ln -s /opt/kenlm /opt/OpenSeq2Seq/ctc_decoder_with_lm/kenlm

WORKDIR /opt/tensorflow

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

RUN ln -s /opt/OpenSeq2Seq/ctc_decoder_with_lm /opt/tensorflow/ctc_decoder_with_lm

RUN bazel build -c opt --copt="-D_GLIBCXX_USE_CXX11_ABI=0"  --copt=-msse4.2 --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --config=cuda  //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so  //ctc_decoder_with_lm:libctc_decoder_with_kenlm.so //ctc_decoder_with_lm:generate_trie

RUN cp /opt/tensorflow/bazel-bin/ctc_decoder_with_lm/*.so /opt/OpenSeq2Seq/ctc_decoder_with_lm/

RUN mkdir /opt/OpenSeq2Seq/language_model
RUN ln -s /data/speech/LM/mozilla-lm.binary /opt/OpenSeq2Seq/language_model/lm.binary
RUN ln -s /data/speech/LM/mozilla-lm.trie   /opt/OpenSeq2Seq/language_model/trie 

ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/opt/OpenSeq2Seq/ctc_decoder_with_lm:/usr/local/lib/python3.5/dist-packages/tensorflow:$LD_LIBRARY_PATH 

#================================================================================

WORKDIR /opt/OpenSeq2Seq


