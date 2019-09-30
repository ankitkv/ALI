"""Some parts of this file are borrowed from https://github.com/9310gaurav/ali-pytorch"""

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from pylego import ops

from ..baseali import BaseALI


class Decoder(nn.Module):

    def __init__(self, latent_size, leak=0.1):
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 256, 4, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leak, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(leak, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leak, inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(leak, inplace=True),

            nn.ConvTranspose2d(32, 32, 5, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(leak, inplace=True),

            nn.ConvTranspose2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(leak, inplace=True),

            nn.ConvTranspose2d(32, 3, 1, bias=False)
        )

        self.output_bias = nn.Parameter(torch.zeros(1, 3, 32, 32))

    def forward(self, input_):
        return torch.tanh(self.main(input_) + self.output_bias)


class Encoder(nn.Module):

    def __init__(self, latent_size, reparameterization=True, leak=0.1):
        super().__init__()
        self.reparameterization = reparameterization
        self.latent_size = latent_size
        if reparameterization:
            latent_size = latent_size * 2

        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 5, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(leak, inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leak, inplace=True),

            nn.Conv2d(64, 128, 4, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(leak, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leak, inplace=True),

            nn.Conv2d(256, 512, 4, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(leak, inplace=True),

            nn.Conv2d(512, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(leak, inplace=True),

            nn.Conv2d(512, latent_size, 1, bias=False)
        )

        self.output_bias = nn.Parameter(torch.zeros(1, latent_size, 1, 1))

    def forward(self, input_):
        output = self.main(input_) + self.output_bias
        if self.reparameterization:
            mean = output[:, :self.latent_size]
            logvar = output[:, self.latent_size:]
            output = ops.reparameterize_gaussian(mean, logvar, self.training)
        return output


class Generator(nn.Module):

    def __init__(self, latent_size, reparameterization=True, leak=0.1, use_sn=True, power_iterations=1):
        super().__init__()
        self.encoder = Encoder(latent_size, reparameterization=reparameterization, leak=leak)
        self.decoder = Decoder(latent_size, leak=leak)

        if use_sn:
            for module in self.modules():
                if hasattr(module, 'weight') and module.weight is not None and not isinstance(module, nn.Embedding):
                    if any(isinstance(module, mtype) for mtype in [nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm1d,
                                                                   nn.InstanceNorm2d]):
                        module.weight.data.fill_(1)
                        module.weight.requires_grad = False
                    else:
                        spectral_norm(module, n_power_iterations=power_iterations)

    def forward(self, q_x):
        q_x = (q_x * 2.0) - 1.0  # scale to [-1, +1]
        q_z = self.encoder(q_x)

        p_z = torch.randn_like(q_z)
        p_x = self.decoder(p_z)

        return p_z, p_x, q_z, q_x

    def visualize(self, x, z):
        if x is not None:
            x = (x * 2.0) - 1.0  # scale to [-1, +1]
            z = self.encoder(x)
        x = self.decoder(z)
        return (x + 1.0) / 2.0


class Discriminator(nn.Module):

    def __init__(self, latent_size, leak=0.1, use_sn=True, power_iterations=1):
        super().__init__()

        self.infer_x = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.LeakyReLU(leak, inplace=True),

            nn.Conv2d(32, 64, 4, stride=2),
            nn.LeakyReLU(leak, inplace=True),

            nn.Conv2d(64, 128, 4),
            nn.LeakyReLU(leak, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2),
            nn.LeakyReLU(leak, inplace=True),

            nn.Conv2d(256, 512, 4),
            nn.LeakyReLU(leak, inplace=True)
        )

        self.infer_z = nn.Sequential(
            nn.Conv2d(latent_size, 512, 1, bias=False),
            nn.LeakyReLU(leak, inplace=True),

            nn.Conv2d(512, 512, 1, bias=False),
            nn.LeakyReLU(leak, inplace=True)
        )

        self.infer_joint = nn.Sequential(
            nn.Conv2d(1024, 1024, 1),
            nn.LeakyReLU(leak, inplace=True),

            nn.Conv2d(1024, 1024, 1),
            nn.LeakyReLU(leak, inplace=True)
        )

        self.final = nn.Conv2d(1024, 1, 1)

        if use_sn:
            for module in self.modules():
                if hasattr(module, 'weight') and module.weight is not None and not isinstance(module, nn.Embedding):
                    if any(isinstance(module, mtype) for mtype in [nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm1d,
                                                                   nn.InstanceNorm2d]):
                        module.weight.data.fill_(1)
                        module.weight.requires_grad = False
                    else:
                        spectral_norm(module, n_power_iterations=power_iterations)

    def forward(self, x, z):
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        output_features = self.infer_joint(torch.cat([output_x, output_z], dim=1))
        output = self.final(output_features)
        return output.view(output.size(0), 1)


class ALIModel(BaseALI):

    def __init__(self, flags, *args, **kwargs):
        generator = Generator(flags.latent_size, reparameterization=flags.reparameterization, leak=flags.leak,
                              use_sn=flags.sn_g, power_iterations=flags.power_iterations)
        discriminator = Discriminator(flags.latent_size, leak=flags.leak, use_sn=flags.sn,
                                      power_iterations=flags.power_iterations)
        super().__init__(flags, generator, discriminator, *args, **kwargs)

    def get_disc_batches(self, forward_ret):
        p_z, p_x, q_z, q_x = forward_ret

        # joint distributions to match
        forward_batch = (p_x, p_z)  # forward
        backward_batch = (q_x, q_z)  # backward

        return forward_batch, backward_batch
