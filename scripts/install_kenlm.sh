set -xe
if [ ! -d "kenlm.tar.gz" ]; then
  wget http://kheafield.com/code/kenlm.tar.gz
fi
tar xf kenlm.tar.gz
cd kenlm
mkdir build
cd build
cmake ..
make -j
cd ..
ln -s  ../kenlm ../ctc_decoder_with_lm/kenlm
