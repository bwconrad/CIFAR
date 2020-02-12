import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from utils import print_and_log

def load_data(config):
    dataset_name = config['dataset']
    print_and_log('\nLoading {} dataset...'.format(dataset_name), config['log'])
    if dataset_name == 'cifar10':
        config['n_classes'] = 10
        return load_cifar10(config)

    else:
        return NotImplementedError('{} is not an available dataset}'.format(dataset_name))


def load_cifar10(config):
    # Define transformations
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]        

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load dataset
    train_dataset = datasets.CIFAR10(config['data_path'], train=True, transform=train_transforms, download=True)
    test_dataset = datasets.CIFAR10(config['data_path'], train=False, transform=test_transforms, download=True)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config['batch_size'], shuffle=True, num_workers=config['workers'], pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = config['batch_size'], shuffle=False, num_workers=config['workers'], pin_memory=True)
    
    return train_loader, test_loader