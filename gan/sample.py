from torchvision import transforms

from gan.BigGANmh import Generator
import gan.utils as utils

def sample_gan(gen, batch_size, truncation, device):
    '''
    Sample a batch of images from the generator
    '''
    z, y = utils.prepare_z_y(batch_size, dim_z=128, nclasses=10, device=device,)    
    images, labels = utils.sample(gen, z, y)  
    return normalize(images), labels

def normalize(images):
    '''
    Normalize generated images the same way as training dataset
    '''
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]  
    normalize = transforms.Normalize(mean, std)

    # Normalize each image in the batch
    for i in range(len(images)):
        images[i] = images[i].add(1.0).mul(0.5).clamp(0.0, 1.0)
        images[i] = normalize(images[i])

    return images