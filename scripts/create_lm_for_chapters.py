import argparse
import os
import pandas as pd

from scripts import preprocess

parser = argparse.ArgumentParser(
    description='Split book into chapters for lm'
)
parser.add_argument('--path', default="/raid/Speech/LibriSpeech-custom/Book_1/pg39906.txt")
parser.add_argument('--wav_csv', default="/raid/Speech/LibriSpeech-custom/Book_1/book1.csv")
parser.add_argument('--ngrams', default=2)
parser.add_argument('--output_path', default="/raid/Speech/LibriSpeech-custom/Book_1/chapters")

args = parser.parse_args()

input_book_path = args.path
ngrams = args.ngrams
book = preprocess.clean_book(input_book_path)
output_path = args.output_path
csv_path = args.wav_csv
files = pd.read_csv(csv_path)["File"]
i = 1
if not os.path.isdir(output_path):
  os.makedirs(output_path)
op_csv=[]
os.system("chmod +x kenlm/build/bin/lmplz")
os.system("chmod +x kenlm/build/bin/build_binary")
for ch in book:
  print("####Processing chapter {}####".format(i))
  op_path = output_path+"/chapter_{}.binary".format(i)
  chapter = "".join(ch)
  with open("/tmp/temp_ch.txt","w") as f:
    f.write(chapter)
  os.system("kenlm/build/bin/lmplz -o {} </tmp/temp_ch.txt >/tmp/temp_ch.arpa".format(ngrams))
  os.system("kenlm/build/bin/build_binary -s /tmp/temp_ch.arpa {}".format(op_path))
  os.remove("/tmp/temp_ch.arpa")
  os.remove("/tmp/temp_ch.txt")
  op_csv.append([files[i-1],op_path])
  i+=1
  print("####Language model for chapter {} completed####".format(i-1))
df = pd.DataFrame(op_csv, columns=["File","lm"])
df.to_csv(os.path.join(output_path+"/language_models.csv"))

