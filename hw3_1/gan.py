import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from . import autoencoder


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.

        modules = []

        modules.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2))
        modules.append(nn.ReLU(inplace=True))
        for channel_in, channel_out in [(32, 128), (128, 256), (256, 256)]:
            modules.append(nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                                  kernel_size=5, padding=2, stride=2, bias=False))
            modules.append(nn.BatchNorm2d(num_features=channel_out, momentum=0.9))
            modules.append(nn.ReLU())

        with torch.no_grad():
            x = torch.randn(1, *in_size)
            h = nn.Sequential(*modules)(x)
            n_features = torch.numel(h) // h.shape[0]

        modules.append(nn.Flatten())
        modules.append(nn.Linear(in_features=n_features, out_features=512, bias=False))
        modules.append(nn.BatchNorm1d(num_features=512, momentum=0.9))
        modules.append(nn.ReLU(True))
        modules.append(nn.Linear(in_features=512, out_features=1))

        self.conv = nn.Sequential(*modules)
        self.eval()


    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        y = self.conv(x)
        return torch.sigmoid(y)


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim
        features_shape = torch.Size((256, 8, 8))
        n_features = torch.FloatTensor(features_shape).numel()
        print(features_shape, n_features)

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        self.decoder = autoencoder.DecoderCNN(in_channels=features_shape[0], out_channels=out_channels)
        self.decoder_fc = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=n_features, bias=False),  # z_dim, n_features
            nn.BatchNorm1d(num_features=n_features, momentum=0.9),
            nn.ReLU(True),
            autoencoder.Unflatten(1, features_shape)
        )

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        samples = Variable(torch.randn(n, self.z_dim), requires_grad=with_grad)
        if with_grad:
            samples = self.forward(samples)
        else:
            with torch.no_grad():
                samples = self.forward(samples)
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        #print(f'z:{z.shape}')
        h = self.decoder_fc(z)
        #print(f'h:{h.shape}')
        x = self.decoder(h)
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    if data_label:
        if label_noise:
            r1, r2 = 1-label_noise, 1+label_noise
            print(r1, r2, (r2-r1))
            labels = torch.rand(len(y_data)) * (r2-r1) + r1
            m = torch.distributions.uniform.Uniform(r1, r2)
            labels = m.sample(y_data.shape)
        else:
            labels = torch.ones_like(y_data)
    else:
        if label_noise:
            r1, r2 = 0-label_noise, 0+label_noise
            labels = torch.rand(len(y_data)) * (r2-r1) + r1
            m = torch.distributions.uniform.Uniform(r1, r2)
            labels = m.sample(y_data.shape)
        else:
            labels = torch.zeros_like(y_data)
    print(labels)

    bce_data = nn.BCEWithLogitsLoss()
    bce_gen = nn.BCEWithLogitsLoss()

    loss_data = bce_data(y_data, Variable(labels))
    r1, r2 = 0-label_noise, 0+label_noise
    print(Variable(torch.rand(len(y_generated)) * (r2-r1) + r1))
    m = torch.distributions.uniform.Uniform(r1, r2)
    labels = m.sample(y_data.shape)
    loss_generated = bce_gen(y_generated, Variable(torch.zeros_like(y_generated)))

    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    if data_label:
        labels = torch.ones_like(y_generated)
    else:
        labels = torch.zeros_like(y_generated)

    bce = nn.BCEWithLogitsLoss()
    loss = bce(y_generated, Variable(labels))
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: DataLoader,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================

    return saved
