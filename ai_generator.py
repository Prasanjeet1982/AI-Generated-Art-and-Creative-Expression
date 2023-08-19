import numpy as np
import tensorflow as tf
from PIL import Image

def generate_ai_art(model_path):
    # Load the PGGAN model
    pg_gan = tf.keras.models.load_model(model_path)

    # Generate an art image
    latent_dim = 512
    random_latent = np.random.randn(1, latent_dim)
    generated_image = pg_gan.predict(random_latent)

    # Post-process the generated image
    generated_image = ((generated_image + 1) * 127.5).astype(np.uint8)
    img = Image.fromarray(generated_image[0])
    return img
