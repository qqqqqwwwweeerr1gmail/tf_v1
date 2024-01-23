
import tensorflow as tf
from tensorflow import keras

class QNetwork(keras.Model):
    def __init__(self, hidden_layers_size, gamma, learning_rate, input_size, num_of_actions):
        super(QNetwork, self).__init__()

        self.dense_layers = [tf.keras.layers.Dense(units=l, activation=tf.nn.relu, kernel_initializer='glorot_uniform') for l in hidden_layers_size]
        self.last_layer = tf.keras.layers.Dense(units=num_of_actions, activation=None, kernel_initializer='glorot_uniform')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self, inputs, actions, future_actions, q_target, r):
        _layer = inputs
        for layer in self.dense_layers:
            _layer = layer(_layer)
        last_layer_output = self.last_layer(_layer)

        gamma = 0.99
        predictions = tf.reduce_sum(last_layer_output * actions, axis=1)
        labels = r + gamma * tf.reduce_sum(q_target * future_actions, axis=1)
        cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=labels, predictions=predictions))
        return cost

# Instantiate the model
model = QNetwork(hidden_layers_size=[256, 128], gamma=0.99, learning_rate=0.001, input_size=10, num_of_actions=4)

# Save the model
model.save('q_network_model')
















