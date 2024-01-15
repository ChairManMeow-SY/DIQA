import VAE
import torch.nn as nn


class DIQAModel(nn.Module):
    def __init__(self, in_features, latent_size, y_size=0):
        super(DIQAModel, self).__init__()

        self.vae = VAE.VAE(in_features, latent_size, y_size)
        # needs a quality assessment layers

    def forward(self, X, y=None, *args, **kwargs):
        mu_prime, mu, log_var = self.vae(X, y)
        return mu_prime

