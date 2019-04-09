# External LM re-scoring

We use Transformer-XL code taken from: https://github.com/kimiyoung/transformer-xl/tree/master/pytorch 

1) Run beam search decoder and dump beams
2) Download checkpoints: [LibriSpeech](https://drive.google.com/a/nvidia.com/file/d/15--z08YNePr8Fgx4cnY4zR37QPZ3ZfMf/view?usp=sharing) and [WSJ](https://drive.google.com/a/nvidia.com/file/d/13D4Hwr_fOd85tkzLxDchEsodlOWdyBhY/view?usp=sharing).  Then have a look at ``run_lm_exp.sh`` for example on how to run re-scoring
