import tensorflow as tf

def create_train_op(optimizer, loss, variables, max_gradient_norm=None, name='training'):
    with tf.variable_scope(name):
        grads_and_vars = optimizer.compute_gradients(loss, variables)
        return optimizer.apply_gradients(
            clip_gradients(grads_and_vars, max_gradient_norm),
            global_step=tf.train.create_global_step())

def clip_gradients(grads_and_vars, max_gradient_norm):
    if max_gradient_norm is None:
        return grads_and_vars

    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    return zip(clipped_gradients, variables)