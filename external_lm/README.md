External Language Model
=======================

Steps to build C++ code
-----------------------

* `mkdir build`
* `cd build`
* `cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..`
* `make`

Here `/path/to/libtorch` should be full path the unzipped LibTorch distribution. 
Download it from here https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip 

You will also need saved traced (on CPU) model and vocabluary files.

You should trace forward pass which will return sentence probability. See example below.
```python
# This is a part which'll be traced.
class InferenceModel(nn.Module):
    def __init__(self, nn_model):
        super(InferenceModel, self).__init__()
        self._nn_model = nn_model
        self.softmax = torch.nn.Softmax(dim=2)


    def forward(self, input):
        input = input.long()
        # input is [time, batch]
        output, hidden = self._nn_model(input)
        # outputs are now [time, batch, vocab]
        prob_distr = self.softmax(output) # [time, batch, vocab]
        # select relevant probability scores
        probs = torch.gather(prob_distr, dim=2, index=input.unsqueeze(-1)).squeeze(-1)
        # take logarithm
        return torch.sum(probs)
```