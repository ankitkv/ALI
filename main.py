import argparse
import glob
import os

from pylego.misc import add_argument as arg
from runners.alirunner import ALIRunner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg(parser, 'name', type=str, required=True, help='name of the experiment')
    arg(parser, 'model', type=str, default='ali.ali', help='model to use')
    arg(parser, 'cuda', type=bool, default=True, help='enable CUDA')
    arg(parser, 'load_file', type=str, default='', help='file to load model from')
    arg(parser, 'save_file', type=str, default='model.dat', help='model save file')
    arg(parser, 'save_every', type=int, default=500, help='save every these many global steps (-1 to disable saving)')
    arg(parser, 'data_path', type=str, default='data')
    arg(parser, 'logs_path', type=str, default='logs')
    arg(parser, 'force_logs', type=bool, default=False)
    arg(parser, 'resume_checkpoint', type=bool, default=True)
    arg(parser, 'learning_rate_g', type=float, default=0.0001, help='Generator learning rate')
    arg(parser, 'learning_rate_d', type=float, default=0.0001, help='Discriminator learning rate')
    arg(parser, 'disc_iters', type=int, default=5, help='no. of discriminator iters before generator update')
    arg(parser, 'lr_decay', type=float, default=0.999, help='learning rate decay')
    arg(parser, 'decay_every', type=int, default=60000, help='decay learning rate every these many training samples')
    arg(parser, 'loss', type=str, default='bce', help='one of: wasserstein, bce')
    arg(parser, 'wasserstein_nodrift', type=float, default=0.0001,
        help='weight to prevent for wasserstein discriminator output from drifting away from 0')
    arg(parser, 'latent_size', type=int, default=64, help='latent dimensions')
    arg(parser, 'leak', type=float, default=0.1, help='LeakyReLU slope')
    arg(parser, 'gp', type=bool, default=True, help='Use the gradient penalty (only if wasserstein loss enabled)')
    arg(parser, 'sn', type=bool, default=True, help='Use spectral norm in the discriminator')
    arg(parser, 'sn_g', type=bool, default=False, help='Use spectral norm in the generator')
    arg(parser, 'grad_norm', type=float, default=5.0, help='gradient norm clipping (-1 to disable)')
    arg(parser, 'batch_size', type=int, default=100, help='batch size')
    arg(parser, 'power_iterations', type=int, default=1, help='power iterations for spectral norm')
    arg(parser, 'reparameterization', type=bool, default=True, help='Use reparameterization for noise')
    arg(parser, 'epochs', type=int, default=50000, help='no. of training epochs')
    arg(parser, 'max_batches', type=int, default=-1, help='max batches per split (if not -1, for debugging)')
    arg(parser, 'print_every', type=int, default=100, help='print losses every these many steps')
    arg(parser, 'gpus', type=str, default='0')
    arg(parser, 'threads', type=int, default=-1, help='data processing threads (-1 to determine from CPUs)')
    arg(parser, 'debug', type=bool, default=False, help='run model in debug mode')
    arg(parser, 'visualize_every', type=int, default=-1,
        help='visualize during training every these many steps (-1 to disable)')
    arg(parser, 'visualize_only', type=bool, default=False, help='epoch visualize the loaded model and exit')
    arg(parser, 'visualize_split', type=str, default='test', help='split to visualize with visualize_only')

    flags = parser.parse_args()
    if flags.threads < 0:
        flags.threads = max(1, len(os.sched_getaffinity(0)) - 1)
    if flags.grad_norm < 0:
        flags.grad_norm = None

    iters = 0
    while True:
        if iters == 4:
            raise IOError("Too many retries, choose a different name.")
        flags.log_dir = '{}/{}'.format(flags.logs_path, flags.name)
        try:
            print('* Creating log dir', flags.log_dir)
            os.makedirs(flags.log_dir)
            break
        except IOError as e:
            if flags.force_logs or flags.resume_checkpoint:
                print('*', flags.log_dir, 'not recreated')
                break
            else:
                print('*', flags.log_dir, 'already exists')
                flags.name = flags.name + "_"
        iters += 1

    flags.save_file = flags.log_dir + '/' + flags.save_file
    if flags.resume_checkpoint:
        existing = glob.glob(flags.save_file + ".*")
        pairs = [(f.rsplit('.', 1)[-1], f) for f in existing]
        pairs = sorted([(int(k), f) for k, f in pairs if k.isnumeric()], reverse=True)
        if pairs:
            print('* Checkpoint resuming is enabled, found checkpoint at', pairs[0][1])
            flags.load_file = pairs[0][1]

    print('Arguments:', flags)
    if flags.visualize_only and not flags.load_file:
        print('! WARNING: visualize_only without load_file!')

    if flags.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpus

    if flags.model.startswith('ali.'):
        runner = ALIRunner
    runner(flags).run(visualize_only=flags.visualize_only, visualize_split=flags.visualize_split, val_split='test')
