echo "=== Downloading datasets ==="
echo "---"

mkdir -p data
cd data

echo "- Downloading WikiText-2 (WT2)"
wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip -q wikitext-2-v1.zip
cd wikitext-2
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt
cd ..

echo "- Downloading WikiText-103 (WT103)"
wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip -q wikitext-103-v1.zip
cd wikitext-103
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt
cd ..

echo "=== Download and pre-processing completed ==="
