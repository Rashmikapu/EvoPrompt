#Seed 1
from utils import *
import os
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt


USE_GPU = True

if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'

class CustomConvNet(tf.keras.Model):
    def __init__(self):
        super(CustomConvNet, self).__init__()

        # Network - conv- batch norm- relu -max_pooling
        # 2 such layers
        # Last layer is FCN- softmax
        num_classes = 10
        input_shape = (32,32,3)
        filter_kernels = [5,7,3]
        channels = [100, 60, 30]

        # Initializer
        initializer = tf.initializers.VarianceScaling(scale=2.0, seed=42)


        # First layer
        # Network - conv- batch norm- relu -max_pooling
        ker_size = np.random.choice(filter_kernels)
        channel = np.random.choice(channels)

        self.conv1 = tf.keras.layers.Conv2D(channel, (ker_size, ker_size), padding='same',
                                   input_shape=input_shape, kernel_initializer=initializer)
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.drop1 = tf.keras.layers.Dropout(0.25)


        # Final connected layer (Dense , FCN)
        self.fcn = tf.keras.layers.Dense(num_classes, activation='softmax',
                                   kernel_initializer=initializer)

        self.flatten = tf.keras.layers.Flatten()

    def call(self, input_tensor, training=False):

            x = self.conv1(input_tensor)
            x = self.batch1(x)
            x = self.relu1(x)
            x = self.maxpool1(x)
            x = self.drop1(x)

            
            x = self.flatten(x)

            x = self.fcn(x)

            return x

def main():
  print_every = 700
  num_epochs = 10

  model = CustomConvNet()

  def model_init_fn():
      return CustomConvNet()

  def optimizer_init_fn():
      learning_rate = 1e-3
      return tf.keras.optimizers.Adam(learning_rate)

  acc, params = train_part34(model_init_fn, optimizer_init_fn, num_epochs=num_epochs, is_training=True)

  return acc, params

if __name__ == "__main__":
    accuracy, model_size = main()
    print(f"Final Accuracy: {accuracy*100}%, Model Size: {model_size}")