import torch
import torch.utils.data as data
from torchvision import transforms, datasets

def load_data(config):
    dataset_name = config['dataset']
    print('\nLoading {} dataset...'.format(dataset_name))
    if dataset_name == 'cifar10':
        config['n_classes'] = 10
        return load_cifar10(config)

    else:
        return NotImplementedError('{} is not an available dataset}'.format(dataset_name))


def load_cifar10(config):
    # Define transformations
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]        

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
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