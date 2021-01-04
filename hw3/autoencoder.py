from typing import List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, Size
from torch.nn import ReLU, BatchNorm2d, Conv2d
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.

        modules = []
        for channel_in, channel_out in [(in_channels, 64), (64, 128), (128, out_channels)]:
            modules.append(Conv2d(in_channels=channel_in, out_channels=channel_out,
                                  kernel_size=5, padding=2, stride=2, bias=False))
            modules.append(BatchNorm2d(num_features=channel_out, momentum=0.9))
            modules.append(ReLU())

        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels=256, out_channels=32):
        super().__init__()

        modules = []

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        modules = []
        for channel_in, channel_out in [(in_channels, in_channels), (in_channels, in_channels//2), (in_channels//2, in_channels//8)]:
            modules.append(
                nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2, output_padding=1,
                                   bias=False))
            modules.append(BatchNorm2d(num_features=channel_out, momentum=0.9))
            modules.append(ReLU())
        modules.append(nn.Conv2d(in_channels=in_channels//8, out_channels=out_channels, kernel_size=5, stride=1, padding=2))
        modules.append(nn.Tanh())

        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        return torch.tanh(self.cnn(h))


class Unflatten(nn.Module):
    def __init__(self, unflattened_size: Size) -> None:
        super(Unflatten, self).__init__()
        self.unflattened_size = unflattened_size

    def forward(self, input: Tensor) -> Tensor:
        return input.view(len(input), *self.unflattened_size)

    def extra_repr(self) -> str:
        return 'unflattened_size={}'.format(self.unflattened_size)


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)
        # TODO: Add more layers as needed for encode() and decode().
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_features, out_features=1024, bias=False),#self.features_shape[0]
            nn.BatchNorm1d(num_features=1024, momentum=0.9),
            nn.ReLU(True)
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=n_features, bias=False), #z_dim, n_features
            nn.BatchNorm1d(num_features=n_features, momentum=0.9),
            nn.ReLU(True),
            Unflatten(self.features_shape)
        )
        # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=1024, out_features=z_dim)
        self.l_var = nn.Linear(in_features=1024, out_features=z_dim)
        self.eval()

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # TODO:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        encode = self.features_encoder(x)
        h = self.encoder_fc(encode)
        mu = self.l_mu(h)
        logvar = self.l_var(h)

        log_sigma2 = torch.exp(logvar * 0.5)
        sample = Variable(torch.randn(len(x), self.z_dim, device=x.device), requires_grad=True)
        z = sample * log_sigma2 + mu

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        h = self.decoder_fc(z)
        x_rec = self.features_decoder(h)
        return torch.tanh(x_rec)

    def sample(self, n):
        device = next(self.parameters()).device
        samples = []
        with torch.no_grad():
            # TODO:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            self.eval()
            for _ in range(n):
                sample = Variable(torch.randn(1, self.z_dim, device=device), requires_grad=True)
                samples.append(self.decode(sample)[0])
            self.train()

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]

        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    # TODO:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.

    data_loss = torch.mean((x - xr) ** 2) / x_sigma2
    kl = torch.sum(z_log_sigma2.exp() + torch.pow(z_mu, 2) - z_log_sigma2 - 1, 1)
    kldiv_loss = torch.mean(kl)
    loss = data_loss + kldiv_loss
    #print(f'loss:{loss}, data_loss:{data_loss}, kldiv_loss:{kldiv_loss}')
    return loss, data_loss, kldiv_loss
