apt-get install -y liblapack-dev gfortran
pip3 install scipy
pip3 install pandas

apt-get install -y llvm-7
ln -s /usr/bin/llvm-config-7 /usr/bin/llvm-config
pip3 install llvmlite
pip3 install Cython
pip3 install librosa
pip3 install nltk==3.2.5

apt-get install -y libfreetype6-dev
pip3 install matplotlib

apt-get install -y portaudio19-dev
pip3 install pyaudio

git clone https://github.com/google/sentencepiece
cd sentencepiece
mkdir build
cd build
apt-get install -y cmake
cmake ..
make -j $(nproc)
make install
ldconfig -v
cd ../..
rm -rf sentencepiece
pip3 install sentencepiece

pip3 install python-speech-features
pip3 install tqdm

apt-get install -y sox
pip3 install sox

pip3 install sacrebleu
