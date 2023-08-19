import tensorflow as tf

def load_pg_gan_model(model_path):
    return tf.keras.models.load_model(model_path)
