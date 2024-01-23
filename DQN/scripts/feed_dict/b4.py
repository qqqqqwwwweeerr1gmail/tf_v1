import tensorflow as tf


def ds_train(new_val_size, new_epochs):
    new_val = (tf.data.Dataset.from_tensor_slices(([16.2, 76.2, 38.4, 11.6, 19.3], [-12, -15, -28, -45, -89]))
               .batch(new_val_size)
               .repeat(new_epochs)
               )
    return new_val


new_val_size = 1
input_size = 1
new_epochs = 2

with tf.variable_scope("dataset"):
    result = ds_train(new_val_size, new_epochs)

with tf.variable_scope("iterator"):
    val_iterate = result.make_initializable_iterator()
    new_iterate_handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(new_iterate_handle,
                                                   val_iterate.output_types,
                                                   val_iterate.output_shapes)


    def next_item():
        new_element = iterator.get_next(name="new_element")
        m, n = tf.cast(new_element[0], tf.float32), new_element[1]
        return m, n

inputs = tf.Variable(tf.zeros(shape=[new_val_size, input_size]), dtype=tf.float32, name="inputs", trainable=False,
                     use_resource=True)
target = tf.Variable(tf.zeros(shape=[new_val_size], dtype=tf.int32), dtype=tf.int32, name="target", trainable=False,
                     use_resource=True)
is_new = tf.placeholder_with_default(tf.constant(False), shape=[], name="new_item_flag")


def new_data(new_val_size, input_size):
    next_inputs, next_target = next_item()
    next_inputs = tf.reshape(next_inputs, shape=[new_val_size, input_size])
    with tf.control_dependencies([tf.assign(inputs, next_inputs), tf.assign(target, next_target)]):
        return tf.identity(inputs), tf.identity(target)


def old_data():
    return inputs, target


next_inputs, next_target = next_item()

inputs, target = tf.cond(is_new, lambda: new_data(new_val_size, input_size), old_data)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    handle_t = sess.run(val_iterate.string_handle())
    sess.run(val_iterate.initializer)
    while True:
        try:
            print(sess.run([inputs, target], feed_dict={new_iterate_handle: handle_t, is_new: False}))
            print(sess.run([inputs, target], feed_dict={new_iterate_handle: handle_t, is_new: False}))
            print(sess.run([inputs, target], feed_dict={new_iterate_handle: handle_t, is_new: False}))
            print(sess.run([inputs, target], feed_dict={new_iterate_handle: handle_t, is_new: False}))
            print(sess.run([inputs, target], feed_dict={new_iterate_handle: handle_t, is_new: True}))
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break