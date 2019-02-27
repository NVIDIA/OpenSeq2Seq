#!/bin/sh
git clone https://github.com/PaddlePaddle/DeepSpeech
mv DeepSpeech/decoders/swig_wrapper.py DeepSpeech/decoders/swig/ctc_decoders.py
mv DeepSpeech/decoders/swig ./decoders
rm -rf DeepSpeech
cd decoders
sed -i "s/\.decode('utf-8')//g" ctc_decoders.py
sed -i 's/\.decode("utf-8")//g' ctc_decoders.py
sed -i "s/name='swig_decoders'/name='ctc_decoders'/g" setup.py
sed -i "s/'swig_decoders'/'ctc_decoders', 'swig_decoders'/g" setup.py
chmod +x setup.sh
./setup.sh
cd ..
