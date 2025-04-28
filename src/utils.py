import numpy as np
from tensorflow.keras.datasets import mnist

def load_federated_mnist(num_clients=4):
    # Split MNIST into num_clients shards
    (x, y), _ = mnist.load_data()
    x = x.astype('float32') / 255.0
    shards = np.array_split(np.arange(len(x)), num_clients)
    data = [(x[s], y[s]) for s in shards]
    return data

# Model building helpers
from tensorflow.keras import layers, Model

def build_generator(latent_dim=100, num_classes=10):
    noise = layers.Input(shape=(latent_dim,))
    label = layers.Input(shape=(1,), dtype='int32')
    label_embed = layers.Embedding(num_classes, latent_dim)(label)
    merged = layers.multiply([noise, layers.Flatten()(label_embed)])
    # ... build Dense->Reshape->Conv2DTranspose blocks ...
    out = layers.Conv2DTranspose(1, 5, activation='tanh')(x)
    return Model([noise, label], out, name='generator')

def build_discriminator(img_shape=(28,28,1), num_classes=10):
    img = layers.Input(shape=img_shape)
    # ... build Conv2D->LeakyReLU blocks ...
    out = layers.Dense(num_classes, activation='softmax')(x)
    return Model(img, out, name='discriminator')

# Defense helper
def loss_based_detection(gen_losses, disc_losses, threshold=1.0):
    ratio = np.mean(gen_losses) / (np.mean(disc_losses) + 1e-8)
    return ratio > threshold
