.. _lstmlm:

LSTMLM
============


Model
~~~~~

This is a word-level language model that uses a basic uni-directional LSTM architecture. To train on the dataset WikiText-103, we use 3 layer LSTM model, each layer with 1024 units and embedding size of 400. We break the data into sequences of length 96.

The biggest problem when training RNN-based language models is overfitting. To reduce overfitting, we use several techniques proposed in the paper `Regularizing and Optimizing LSTM Language Models (Smerity et al., 2017) <https://arxiv.org/pdf/1708.02182.pdf>`.

* weights tied: share the same variables between the embedding matrix with the kernel output project layer after the RNN. It means that the LSTM cell at the last layer has the same number of hidden units as the embedding size. Due to the huge vocab size, this technique reduces the number of parameters by 100M.
* random start: According to Smerity et al., if we always break the training data into sequences of length 96, any element at the index divisible by 96 "never has any elements to backprop into". To avoid this, we randomly choose the start token index to be between [0, 95].
* embedding dropout: we use dropout with the embedding matrix (which is also the kernel for the output projection layer) 
* LSTM dropout: we use various dropout methods with LSTM cells. For more information on these different techniques of dropout, this is `an excellent blog post <https://medium.com/@bingobee01/a-review-of-dropout-as-applied-to-rnns-72e79ecd5b7b>`. The cell in the last layer uses a different keep probability because with weight tied, it often has a much smaller number of hidden units.
	- input_weight_keep_prob: keep probability for dropout of W (kernel used to multiply with the input tensor)
	- recurrent_weight_keep_prob: keep probability for dropout of U (kernel used to multiply with last hidden state tensor)
	- recurrent_keep_prob: keep probability for dropout when applying tanh on the input transform 
	- encoder_dp_input_keep_prob: keep probability for dropout on input of a LSTM cell in the layer which is not the last layer
	- encoder_dp_output_keep_prob: keep probability for dropout on output of a LSTM cell in the layer which is not the last layer
	- encoder_last_input_keep_prob: like encoder_dp_input_keep_prob but for the cell in the last layer
	- encoder_dp_output_keep_prob: like encoder_dp_output_keep_prob but for the cell in the last layer

The model has 123.7M tokens. Using single-precision training on 4 GPUs, it takes 36 hours to get to the perplexity of 51, and double that amount of time to get the perplexity of 48.6. Using mixed-precision, we're able to double the batch size and reduce the training time by half.