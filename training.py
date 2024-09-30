import numpy as np

# Set hyperparameters
epochs = 1000
batch_size = 100
noise_dim = 4  # Same as the number of features (in real time it can be 100s)

for epoch in range(epochs):
    idx = np.random.randint(0, X.shape[0], batch_size)
    real_sales = y[idx]

    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    generated_sales = generator.predict(noise)

    # Real and fake labels
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(real_sales, real_labels)
    d_loss_fake = discriminator.train_on_batch(generated_sales, fake_labels)

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    g_loss = gan.train_on_batch(noise, real_labels)

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Discriminator Loss: {d_loss_real[0]}, Generator Loss: {g_loss}")