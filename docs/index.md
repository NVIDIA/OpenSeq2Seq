## Requirements
* Python 3.6
* Tensorflow r1.2 or r1.3 (with GPU support)
* NLTK v3.2.3+

## Unit tests
Checkout the code and make sure the following test pass:
You should see **OK** after each test

```
./create_toy_data.sh
python -m unittest test/data_layer_tests.py
python -m unittest test/model_tests.py
```

# Contents
* [Getting Started](Getting-started.md)
* [Toy Data Example](Toy-data-example.md)
* [Training German to English translator](Training-German-to-English-translator.md)
* [Distributed training using Horovod](Distributed-training.md)
* [Models and Recepies](Models-and-Recepies.md)
* [Question Answering](https://github.com/NVIDIA/OpenSeq2Seq/blob/master/QuestionAnswering/README.md) (related project)

### Authors
[Oleksii Kuchaiev](https://github.com/okuchaiev) and [Siddharth Bhatnagar](https://github.com/siddharthbhatnagar) (internship work @ NVIDIA)

Contributions are welcome!

## Related resources
* [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt)
* [OpenNMT (Torch)](http://opennmt.net/)
* [OpenNMT (Pytorch)](https://github.com/OpenNMT/OpenNMT-py)
* [Tf-seq2seq](https://github.com/google/seq2seq)
* [Moses](http://www.statmt.org/moses/)

## References
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906)
* [Effective approaches to attention-based neural machine translation](https://arxiv.org/abs/1508.04025)
