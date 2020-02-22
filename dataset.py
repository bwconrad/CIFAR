import torch
import torch.utils.data as data
from torchvision import transforms, datasets
import torch.utils.data as data 

from utils import print_and_log
from augments import Cutout, CIFAR10Policy, RandAugment

def load_data(config):
    '''
    Load dataset by name
    '''
    dataset_name = config['dataset']
    print_and_log('\nLoading {} dataset...'.format(dataset_name), config['log'])
    if dataset_name == 'cifar10':
        config['n_classes'] = 10
        return load_cifar10(config)
    else:
        raise NotImplementedError('{} is not an available dataset}'.format(dataset_name))


def load_cifar10(config):
    '''
    Load CIFAR10 dataset with specified transformations into a train and test dataloader
    '''

    # Define transformations
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]        

    
    if config['transforms'] == 'standard':
        print_and_log("Using Standard data transformations.", config['log'])
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    elif config['transforms'] == 'cutout':
        print_and_log("Using Cutout data transformations.", config['log'])
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean, std)])

    elif config['transforms'] == 'autoaugment':
        print_and_log("Using AutoAugment data transformations.", config['log'])
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    
    elif config['transforms'] == 'autoaugment_cutout':
        print_and_log("Using AutoAugment and Cutout data transformations.", config['log'])
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean, std)])

    elif config['transforms'] == 'randaugment':
        print_and_log("Using RandAugment data transformations.", config['log'])
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(n=config['randaug_N'], m=config['randaug_M']),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    elif config['transforms'] == 'randaugment_cutout':
        print_and_log("Using RandAugment and Cutout data transformations.", config['log'])
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(n=config['randaug_N'], m=config['randaug_M']),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean, std)])
    else:
        print_and_log("Using no data transformations.", config['log'])
        train_transforms = transforms.Compose([
            transforms.ToTensor()])

    if config['transforms'] == 'none':
        test_transforms = transforms.Compose([
            transforms.ToTensor()])

    else:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    # Load dataset
    train_dataset = datasets.CIFAR10(config['data_path'], train=True, transform=train_transforms, download=True)
    test_dataset = datasets.CIFAR10(config['data_path'], train=False, transform=test_transforms, download=True)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config['batch_size'], shuffle=True, num_workers=config['workers'], pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = config['batch_size'], shuffle=False, num_workers=config['workers'], pin_memory=True)
    
    return train_loader, test_loader

