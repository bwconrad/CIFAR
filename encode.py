import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
import numpy as np
import os
from tqdm import tqdm
from imageio import imsave
import datetime
import pickle


import AutoGAN.models.autogan_cifar10_a
from models.vgg import VGG19
from dataset import load_data
from utils import load_config, print_and_log
from loss import PerceptualLoss, StyleLoss
import lpips

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True

# Load config file
config = load_config()


# Create directories
config['output_path'] = "{}encode_{}/".format(config['output_path'], str(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))
if not os.path.exists(config['output_path']):
    os.makedirs(config['output_path'])
if not os.path.exists(config['data_path']):
    os.makedirs(config['data_path'])

# Create log file
config['log'] = config['output_path'] + 'log.txt'
print_and_log(config, config['log'], printOut=False)

# Load data
train_loader, test_loader = load_data(config)

# Setup network
gen_net = AutoGAN.models.autogan_cifar10_a.Generator()
gen_net.to(device)

# Load weights
checkpoint = torch.load(config['gan_weights'])
if 'avg_gen_state_dict' in checkpoint:
    gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
    epoch = checkpoint['epoch']
    print('=> loaded checkpoint {} (epoch {})'.format(config['gan_weights'], epoch))
else:
    gen_net.load_state_dict(checkpoint)
    print('=> loaded checkpoint {}'.format(config['gan_weights']))

# Freeze weights
for param in gen_net.parameters():
    param.requires_grad = False
gen_net = gen_net.eval()

# Setup loss
criterionMSE = nn.MSELoss()
criterionLPIPS = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

# Encode
latents = []
for i, (real, _) in enumerate(train_loader):
    real =real.to(device)

    z = torch.FloatTensor(np.random.normal(0, 1, (config['batch_size'], config['z_dim']))).to(device)
    optimizer = optim.Adam([z.requires_grad_()], lr=config['lr'], betas=(config['beta1'], config['beta2'])) 

    for step in range(config['steps']):
        optimizer.zero_grad()

        # Pass latent through generator and normalize output
        img = gen_net(z).add(1.0).mul(0.5).clamp(0.0, 1.0)

        mse_loss = criterionMSE(img, real)
        percept_loss = criterionLPIPS.forward(img, real, normalize=False)
        loss = percept_loss 

        loss.backward()
        optimizer.step()
        
    print_and_log("{} Loss: {} MSE: {} Percept: {}".format(i, loss.item(), mse_loss, percept_loss.item()), config['log'])
        
    latents.append(z.tolist())
    #img = img.mul_(255.0).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    #real = real.mul_(255.0).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    #imsave(config['output_path']+str(i)+'real.png', real[0])
    #imsave(config['output_path']+str(i)+'gen.png', img[0])


with open(config['output_path']+'latents_train.pth', 'wb') as f:
    pickle.dump(latents, f)