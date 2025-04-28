import argparse
import tensorflow as tf
from utils import load_federated_mnist, build_generator, build_discriminator, loss_based_detection

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()

def train(config):
    # Load config (e.g., learning rates, epochs, attack parameters)
    data = load_federated_mnist(config['num_clients'])
    G = build_generator(config['latent_dim'], config['num_classes'])
    D = build_discriminator(config['img_shape'], config['num_classes'])
    # optimizers, loss functions
    for epoch in range(config['epochs']):
        for client_data in data:
            x_real, y_real = client_data
            # Optionally inject label-flipping attack
            # Train discriminator then generator
            # Compute losses
            # Detect if malicious client by loss ratio
            if loss_based_detection(gen_losses, disc_losses):
                # Apply defense: adaptive averaging / set weights
                pass
        if epoch % config['save_interval'] == 0:
            G.save(f'results/checkpoints/G_epoch{epoch}.h5')
            D.save(f'results/checkpoints/D_epoch{epoch}.h5')

if __name__ == '__main__':
    args = parse_args()
    import yaml
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    train(config)
