import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from PIL import Image
import os

# Set the paths to your dataset
data_dir = 'path_to_dataset'  # Replace with the path to your dataset
train_dir = os.path.join(data_dir, 'train')

# Set other training parameters
image_size = (128, 128)
batch_size = 32
latent_dim = 512
epochs = 1000

# Load and preprocess the dataset
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=True
)

# Build the generator model
generator_input = Input(shape=(latent_dim,))
x = Dense(128 * 16 * 16)(generator_input)
x = tf.keras.layers.LeakyReLU()(x)
x = Reshape((16, 16, 128))(x)
x = Conv2DTranspose(128, 4, strides=2, padding='same')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = Conv2DTranspose(128, 4, strides=2, padding='same')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = Conv2DTranspose(128, 4, strides=2, padding='same')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = Conv2D(3, 3, padding='same', activation='tanh')(x)
generator = Model(generator_input, x)

# Build and compile the discriminator model
discriminator_input = Input(shape=image_size + (3,))
x = Conv2D(128, 3)(discriminator_input)
x = tf.keras.layers.LeakyReLU()(x)
x = Conv2D(128, 4, strides=2)(x)
x = tf.keras.layers.LeakyReLU()(x)
x = Conv2D(128, 4, strides=2)(x)
x = tf.keras.layers.LeakyReLU()(x)
x = Conv2D(128, 4, strides=2)(x)
x = tf.keras.layers.LeakyReLU()(x)
x = Flatten()(x)
x = Dense(1, activation='sigmoid')(x)
discriminator = Model(discriminator_input, x)
discriminator.compile(optimizer=Adam(lr=0.0008, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Freeze the discriminator during generator training
discriminator.trainable = False

# Build and compile the GAN model
gan_input = Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan.compile(optimizer=Adam(lr=0.0004, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Training loop
for epoch in range(epochs):
    for step in range(len(train_generator)):
        # Generate random latent vectors
        random_latents = np.random.randn(batch_size, latent_dim)

        # Generate fake images using the generator
        generated_images = generator.predict(random_latents)

        # Train the discriminator on real and fake images
        real_images = next(train_generator)
        discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        discriminator_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

        # Train the generator to fool the discriminator
        random_latents = np.random.randn(batch_size, latent_dim)
        generator_loss = gan.train_on_batch(random_latents, np.ones((batch_size, 1)))

        # Print progress
        print(f"Epoch {epoch + 1}, Step {step + 1}/{len(train_generator)} - D Loss: {discriminator_loss[0]}, G Loss: {generator_loss[0]}")

    # Save generated images at the end of each epoch
    generated_images = (generator.predict(random_latents) + 1) * 0.5 * 255
    for i in range(batch_size):
        img = Image.fromarray(generated_images[i].astype(np.uint8))
        img.save(f"generated_images/epoch_{epoch + 1}_image_{i}.png")

# Save the generator model
generator.save("pggan_generator.h5")
