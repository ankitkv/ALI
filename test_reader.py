from pylego.misc import save_comparison_grid

from readers.cifar10 import CIFAR10Reader


if __name__ == '__main__':
    reader = CIFAR10Reader('data')
    for i, batch in enumerate(reader.iter_batches('train', 256, max_batches=5)):
        img = batch[0].numpy()
        print(img.shape, img.min(), img.max())
        save_comparison_grid('seq%d.png' % i, img, border_shade=0.75)
