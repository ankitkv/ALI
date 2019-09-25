import collections

import numpy as np
import torch

from pylego import misc

from models.baseali import BaseALI
from .basealirunner import BaseALIRunner


class ALIRunner(BaseALIRunner):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(flags, BaseALI)

    def run_batch(self, batch, train=False):
        batch = self.model.prepare_batch(batch[0])
        loss, g_loss, d_loss = self.model.run_loss(batch)
        if train:
            self.model.train(loss, clip_grad_norm=self.flags.grad_norm)

        assert not np.isnan(g_loss)
        assert not np.isnan(d_loss)
        return collections.OrderedDict([('g_loss', g_loss), ('d_loss', d_loss)])

    def post_epoch_visualize(self, epoch, split):
        print('* Visualizing', split)
        fname = self.flags.log_dir + '/vis_{}_{}.png'.format(self.model.get_train_steps(), split)
        if split == 'train':
            z = torch.randn(self.flags.batch_size, self.flags.latent_size, 1, 1)
            z = self.model.prepare_batch(z)
            x = self.model.run_batch([None, z], visualize=True)
            vis_data = x.cpu().numpy()
        else:
            batch = next(self.reader.iter_batches(split, self.batch_size, shuffle=True, partial_batching=True,
                                                  threads=self.threads, max_batches=1))
            batch = self.model.prepare_batch(batch[0])
            x = self.model.run_batch([batch, None], visualize=True)
            y = batch.cpu().numpy()[None, ...]
            x = x.cpu().numpy()[None, ...]
            vis_data = np.concatenate([y, x], axis=0)
            vis_data = np.swapaxes(vis_data, 0, 1).reshape(-1, *x.shape[2:])

        misc.save_comparison_grid(fname, vis_data, border_shade=0.5, retain_sequence=True)
        print('* Visualizations saved to', fname)

    def run(self, train_split='train', val_split='val', test_split='test', visualize_only=False,
            visualize_split='test'):
        """Run the main training loop with validation and a final test epoch, or just visualization on the
        test epoch."""
        epoch = -1
        if visualize_only:
            self.model.set_train(False)
            self.post_epoch_visualize(epoch, train_split)
            self.post_epoch_visualize(epoch, visualize_split)
        else:
            for epoch in range(self.epochs):
                self.run_epoch(epoch, train_split, train=True)
                if val_split:
                    self.post_epoch_visualize(epoch, val_split)
