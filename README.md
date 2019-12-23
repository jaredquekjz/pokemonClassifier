# PokemonClassifier in Tensorflow 2.0

The program trains 4 DNNs that are in turn stack-generalised to achieve a final accuracy of ~75-85%.

The first 3 intermediate DNNs stacks Dense layers with the frozen convolutional layers (pre-trained features) from MobileNet, ResNet and VGG16. The 4th DNN is adapted from an example from Francois Chollet.

This program is developed as part of an experiment in metalearning.

Pokemon data set includes 598 images split into 10 classes adapted from:
https://github.com/rileynwong/pokemon-images-dataset-by-type
