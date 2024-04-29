from __future__ import absolute_import
import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from preprocess import get_data

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.learning_rate = 1e-3
        self.batch_size = 32
        self.num_classes = 26  

        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.output_layer(x)
        return x

    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def train(model, train_inputs, train_labels):
    train_size = train_inputs.shape[0]
    indices = tf.range(start=0, limit=train_size, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    
    train_inputs = tf.gather(train_inputs, shuffled_indices)
    train_labels = tf.gather(train_labels, shuffled_indices)
    
    total_loss = 0
    batches = 0
    
    for start_idx in range(0, train_size, model.batch_size):
        end_idx = start_idx + model.batch_size
        batch_inputs = train_inputs[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]
        
        with tf.GradientTape() as tape:
            logits = model.call(batch_inputs)
            loss = model.loss(logits, batch_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        total_loss += loss.numpy()
        batches += 1

    average_loss = total_loss / batches
    return average_loss

def main():
    print("Main started")
    try:
        train_inputs, train_labels, _, _ = get_data(59)  
        print("Data loaded:", train_inputs.shape, train_labels.shape)

        model = Model()
        print("Model initialized")
        epochs = 10
        for epoch in range(epochs):
            average_loss = train(model, train_inputs, train_labels)
            print(f'Epoch {epoch + 1} completed with average loss: {average_loss}')
    except Exception as e:
        print("An error occurred:", e)



if __name__ == '__main__':
    print("Script started")
    main()

