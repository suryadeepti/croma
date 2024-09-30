import tensorflow as tf
from tensorflow.keras import layers

# Generator model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=4),  # Input is random noise + features
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='linear')  #output layer
    ])
    return model

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(512, activation='relu', input_dim=4),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output is real/fake
    ])
    return model

# Build generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Compile the discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Combined GAN model
discriminator.trainable = False
gan_input = layers.Input(shape=(4,))
generated_sales = generator(gan_input)
gan_output = discriminator(generated_sales)
gan = tf.keras.Model(gan_input, gan_output)

gan.compile(optimizer='adam', loss='binary_crossentropy')

# Show model architecture
generator.summary()
discriminator.summary()