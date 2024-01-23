import tensorflow as tf

state_size = 100  # For example

inputs = tf.keras.Input(shape=(None, state_size))

dense_layer = tf.keras.layers.Dense(32, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(dense_layer)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

we = model.weights
print(we)

la =model.layers
print(la)





























