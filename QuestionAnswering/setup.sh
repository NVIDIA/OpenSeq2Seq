# Download SQuAD data
mkdir datasets && cd datasets
mkdir SQuAD && cd SQuAD
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

# Data processing for SQuAD data
cd ../..
python3 data_preprocessing.py --file datasets/SQuAD/train-v1.1.json --mode train
python3 data_preprocessing.py --file datasets/SQuAD/dev-v1.1.json --mode dev

# Download evaluation script
wget -O eval_SQuAD.py https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/

# Download GloVe script
git clone https://github.com/stanfordnlp/GloVe.git
cd GloVe && make
cp ../TrainGloVe.sh .

# Train SQuAD embedding
cp ../datasets/SQuAD/text.txt .
./TrainGloVe.sh
cp vectors.txt ../datasets/SQuAD/SQuAD.27K.64d.txt
rm cooccurrence.bin cooccurrence.shuf.bin vectors.bin vectors.txt vocab.txt text.txt