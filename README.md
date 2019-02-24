This repository contains two ANN implemented using PyTorch for classifying images from [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
## Models:
- Multilayer perceptron
  <br>Accuracy on test-set: 53% with num_epochs = 6, learning_rate = 0.01, weight_decay = 1e-5
- ResNet-like CNN
<br>Accuracy on test-set: 79%  with num_epochs = 9, learning_rate = 0.01, weight_decay = 1e-5
## How to use:
train.py --model MLP|ResNet
#### Optional parameters:
--num_epochs, default: 2<br>
--learning_rate, default: 0.01<br>
--weight_decay, default: 1e-5
