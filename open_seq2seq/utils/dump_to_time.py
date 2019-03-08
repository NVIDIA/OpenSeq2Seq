import pickle
import numpy as np
import argparse
import sys
args = sys.argv[1:]
parser = argparse.ArgumentParser(description='Experiment parameters')
parser.add_argument("--dumpfile", required=False,type=str,default="/raid/Speech/dump.pkl",
                    help="Path to the configuration file")
parser.add_argument("--blank_index",type=int, default=0, help="Index of blank char")
args = parser.parse_args(args)
dump = pickle.load(open(args.dumpfile,"rb"))
results = dump["results"]
lettermap = dump["dictionary"]
stride = dump["stride"]
scale = dump["scale"]
def ctc_greedy_decoder(logits,wordmap):
    prev_idx = -1
    output = []
    ts = []
    for i,l in enumerate(logits):
        idx = np.argmax(l)
        if(idx!=28 and prev_idx!=idx):
            if(len(output)==0):
                ts.append(stride*scale*i)
            else:
                if(output[-1]==" "):
                    ts.append(stride*scale*i)
            output+=wordmap[idx]

        prev_idx=idx

    output = "".join(output)
    return output,ts
for r in results:
    op,ts=ctc_greedy_decoder(results[r],lettermap)
    print(op)
    print(ts)