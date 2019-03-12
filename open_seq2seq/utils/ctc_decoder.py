import numpy as np
def ctc_greedy_decoder(logits,wordmap,step_size,blank_idx,start_shift,end_shift):
    prev_idx = -1
    output = []
    start = []
    end = []
    lst_letter = -1
    for i,l in enumerate(logits):
        idx = np.argmax(l)
        if(idx!=blank_idx and prev_idx!=idx):
            if(len(output)==0):
                start.append(step_size*i+start_shift)
            else:
                if(output[-1]==" "):
                    start.append(max(step_size*i+start_shift,end[-1]))
            output+=wordmap[idx]
            if output[-1]==" ":
                end.append(step_size*lst_letter+end_shift)
            lst_letter=i
        prev_idx=idx
    end.append(step_size*lst_letter+end_shift)
    output = "".join(output)
    output = output.strip(" ")
    return output,start,end