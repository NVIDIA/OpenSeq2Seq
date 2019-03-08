import pickle
import numpy as np
import argparse
import sys
args = sys.argv[1:]
parser = argparse.ArgumentParser(description='Experiment parameters')
parser.add_argument("--dumpfile", required=True,type=str,default="/raid/Speech/dump.pkl",
                    help="Path to the configuration file")
parser.add_argument("--blank_index",type=int, default=-1, help="Index of blank char")
args = parser.parse_args(args)
dump = pickle.load(open(args.dumpfile,"rb"))
blank_idx = args.blank_index
results = dump["logits"]
vocab = dump["vocab"]
step_size = dump["step_size"]
if blank_idx==-1:
    blank_idx=len(vocab)
def ctc_greedy_decoder(logits,wordmap):
    prev_idx = -1
    output = []
    start = []
    end = []
    for i,l in enumerate(logits):
        idx = np.argmax(l)
        if(idx!=28 and prev_idx!=idx):
            if(len(output)==0):
                start.append(step_size*(i-1))
            else:
                if(output[-1]==" "):
                    start.append(step_size*(i-1))
            output+=wordmap[idx]
            if output[-1]==" ":
                end.append(step_size*(i-2))
        prev_idx=idx

    output = "".join(output)
    return output,start,end

if __name__ == '__main__':
    for r in results:
        letters, starts, ends =ctc_greedy_decoder(results[r],vocab)
        print(letters.starts, ends)