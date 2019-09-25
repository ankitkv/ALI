from abc import abstractmethod
import glob
import pathlib
import sys

import torch
from torch import autograd, nn, optim
from torch.nn import functional as F

from pylego.model import Model


if sys.version_info.minor < 7:
    class nullcontext():
        def __enter__(self):
            return None
        def __exit__(self, *excinfo):
            pass
else:
    from contextlib import nullcontext


class BaseALI(Model):

    def __init__(self, flags, generator, discriminator, *args, **kwargs):
        self.flags = flags
        self.epoch_size = flags.decay_every // flags.batch_size
        self.disc = discriminator

        g_optimizer = optim.Adam(generator.parameters(), lr=flags.learning_rate_g, betas=(0.0, 0.9))
        self.d_optimizer = optim.Adam(self.disc.parameters(), lr=flags.learning_rate_d, betas=(0.0, 0.9))
        self.scheduler_g = optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=flags.lr_decay)
        self.scheduler_d = optim.lr_scheduler.ExponentialLR(self.d_optimizer, gamma=flags.lr_decay)

        self.disc_iters = flags.disc_iters

        kwargs['optimizer'] = g_optimizer
        super().__init__(model=generator, *args, **kwargs)
        self.disc.to(self.device)

        self.g_loss = 0.0
        self.d_loss = 0.0
        if flags.load_file:
            self.load(flags.load_file)

    @abstractmethod
    def get_disc_batches(self, forward_ret):
        """
        Return (backward_batch, forward_batch) for ALI's discriminator.
        """
        pass

    def train_disc(self):
        """
        True if discriminator should be trained, False if generator should be trained.
        """
        return (self.get_train_steps() + 1) % (self.disc_iters + 1) != 0

    def loss_function(self, forward_ret, labels=None):
        if self.is_training():
            forward_batch, backward_batch = self.get_disc_batches(forward_ret)
            if self.debug:
                debug_context = autograd.detect_anomaly()
            else:
                debug_context = nullcontext()
            with debug_context:
                d_ps = self.disc(*forward_batch)
                d_qs = self.disc(*backward_batch)

            if self.train_disc():
                if self.flags.loss == 'wasserstein':
                    if self.flags.gp:
                        grad_penalty = self.gradient_penalty(backward_batch, forward_batch, context=debug_context)
                    else:
                        grad_penalty = 0
                    loss = -d_ps.mean() + d_qs.mean() + (10.0 * grad_penalty) + self.flags.wasserstein_nodrift * \
                        ((d_ps + d_qs)**2).mean()

                else:
                    loss = (F.binary_cross_entropy_with_logits(d_ps, torch.ones_like(d_ps)) +
                            F.binary_cross_entropy_with_logits(d_qs, torch.zeros_like(d_qs)))
                self.d_loss = loss.item()
            else:
                if self.flags.loss == 'wasserstein':
                    loss = d_ps.mean() - d_qs.mean()
                else:
                    loss = (F.binary_cross_entropy_with_logits(d_ps, torch.zeros_like(d_ps)) +
                            F.binary_cross_entropy_with_logits(d_qs, torch.ones_like(d_qs)))
                self.g_loss = loss.item()

            g_loss = self.g_loss
            d_loss = self.d_loss
        else:
            loss, g_loss, d_loss = 0.0, 0.0, 0.0

        return loss, g_loss, d_loss

    def gradient_penalty(self, data1, data2, context=None):
        batch_size = data1[0].size(0)
        epsilon = torch.rand(batch_size, 1, dtype=data1[0].dtype, device=data1[0].device)

        interpolation = []
        for d1, d2 in zip(data1, data2):
            if len(d1.size()) == 5:  # video
                eps = epsilon[..., None, None, None]
            elif len(d1.size()) == 4:  # image
                eps = epsilon[..., None, None]
            else:
                eps = epsilon
            interpolation.append((eps * d1.data + (1 - eps) * d2.data).requires_grad_())

        if context is None:
            context = nullcontext()
        with context:
            interpolation_logits = self.disc(*interpolation)
        grad_outputs = torch.ones_like(interpolation_logits)

        gradients = autograd.grad(outputs=interpolation_logits, inputs=interpolation, grad_outputs=grad_outputs,
                                  create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return ((gradients_norm - 1) ** 2).mean()

    def train(self, loss, clip_grad_norm=None):
        assert self.is_training()
        if self.train_disc():
            optimizer = self.d_optimizer
            model = self.disc
        else:
            optimizer = self.optimizer
            model = self.model

        optimizer.zero_grad()
        if self.debug:
            debug_context = autograd.detect_anomaly()
        else:
            debug_context = nullcontext()
        with debug_context:
            loss.backward()
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        if (self.get_train_steps() + 1) % self.epoch_size == 0:
            self.scheduler_g.step()
            self.scheduler_d.step()
        self.increment_train_steps()

    def run_batch(self, data, visualize=False):
        """If visualize is True, a visualize method of the model module is called."""
        if not isinstance(data, list) and not isinstance(data, tuple):
            data = [data]
        if self.is_training() and not self.train_disc():
            context = nullcontext()
        else:
            context = torch.no_grad()
        if self.debug:
            debug_context = autograd.detect_anomaly()
        else:
            debug_context = nullcontext()
        with context, debug_context:
            if not visualize:
                return self.model(*data)
            else:
                return self.model.visualize(*data)

    def initialize(self, load_file):
        '''Overriding: do not load file during superclass initialization, we do it manually later in init'''
        pass

    def load(self, load_file):
        """Load a model from a saved file."""
        print("* Loading model from", load_file, "...")
        (m_state_dict, d_state_dict, o_state_dict, do_state_dict, sched_state_dict, dsched_state_dict,
         train_steps, g_loss, d_loss) = torch.load(load_file)
        self.model.load_state_dict(m_state_dict)
        self.disc.load_state_dict(d_state_dict)
        self.optimizer.load_state_dict(o_state_dict)
        self.d_optimizer.load_state_dict(do_state_dict)
        self.scheduler_g.load_state_dict(sched_state_dict)
        self.scheduler_d.load_state_dict(dsched_state_dict)
        self.train_steps = train_steps
        self.g_loss = g_loss
        self.d_loss = d_loss
        print("* Loaded model from", load_file)

    def save(self, save_file):
        "Save model to file."
        save_fname = save_file + "." + str(self.train_steps)
        print("* Saving model to", save_fname, "...")
        existing = glob.glob(save_file + ".*")
        pairs = [(f.rsplit('.', 1)[-1], f) for f in existing]
        pairs = sorted([(int(k), f) for k, f in pairs if k.isnumeric()], reverse=True)
        for _, fname in pairs[self.max_save_files - 1:]:
            pathlib.Path(fname).unlink()

        save_objs = [self.model.state_dict(), self.disc.state_dict(), self.optimizer.state_dict(),
                     self.d_optimizer.state_dict(), self.scheduler_g.state_dict(), self.scheduler_d.state_dict(),
                     self.train_steps, self.g_loss, self.d_loss]
        torch.save(save_objs, save_fname)
        print("* Saved model to", save_fname)

    def set_train(self, train):
        if train:
            self.model.train()
            self.disc.train()
        else:
            self.model.eval()
            self.disc.eval()
