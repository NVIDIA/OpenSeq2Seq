## Requirements
* Python 2.7/3.6
* Tensorflow r1.2 or later (with GPU support)
* NLTK v3.2.3+

## Unit tests
Checkout the code and make sure the following test pass:
You should see **OK** after each test
```
./create_toy_data.sh
For Python2 run:
python -m unittest test.data_layer_tests
python -m unittest test.model_tests

For Python3 run:
python -m unittest test/data_layer_tests.py
python -m unittest test/model_tests.py