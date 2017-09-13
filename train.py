import click
import torch

from dataset import ImageFolderDataset
from net import *

torch.backends.cudnn.benchmark = True

@click.command()
@click.option('--name', type=str)
@click.option('--dataset_root', default='../lsun-point')
@click.option('--image_size', default=(404, 404), type=(int, int))
@click.option('--epochs', default=20, type=int)
@click.option('--batch_size', default=1, type=int)
@click.option('--workers', default=6, type=int)
@click.option('--resume', type=click.Path(exists=True))
def main(name, dataset_root, image_size, epochs, batch_size, workers):
    print('===> Prepare dataloader')
    dataset_args = {'root': dataset_root, 'target_size': image_size}
    loader_args = {'num_workers': workers, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
    	dataset=ImageFolderDataset(phase='train', **dataset_args),
    	batch_size=batch_size, **loader_args
    )
    validate_loader = torch.utils.data.DataLoader(
    	dataset=ImageFolderDataset(phase='validate', **dataset_args),
    	batch_size=batch_size, **loader_args
    )
    print('===> Prepare model')
    net = Net(name='new_mdl_testing1', pretrained=True)
    print('===> Start Training')
    net.train(
    	train_loader=train_loader,
    	validate_loader=validate_loader,
    	epochs=epochs
    )
    net.evaluate(data_loader=validate_loader, prefix='')
if __name__ == '__main__':
    main()
