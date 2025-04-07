#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class HandwrittenClassifier(object):
    def __init__(
        self,
        model_path='model/handwritten_classifier/handwritten_classifier_kieuvt212.hdf5',
    ):
        # Load the Keras model
        self.model = tf.keras.models.load_model(model_path)

        # Get the input shape (excluding the batch dimension)
        self.input_shape = self.model.input_shape[1:3]

    def __call__(
        self,
        image,
    ):
        print(self.model.input_shape)
        print(image.shape)
        # Check if the input image shape matches the expected shape
        assert image.shape == self.input_shape, \
            f"Input image shape {image.shape} does not match expected shape {self.input_shape}"

        # Prepare the input data by adding a batch dimension
        # input_data = np.expand_dims(image, axis=0).astype(np.float32)

        # Make the prediction
        
        result = self.model.predict(image.reshape(1,28,28,1))[0]
        print(result)
        # Get the index of the result with the highest probability
        result_index = np.argmax(result)

        return result_index
