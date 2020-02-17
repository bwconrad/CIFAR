import torch
import torchvision
import numpy as np
import os
from tqdm import tqdm
from imageio import imsave
import sys

import AutoGAN.models.autogan_cifar10_a
sys.path.insert(0, 'AutoGAN/')
import cfg
import models

def interp(x0, x1, num_midpoints):
  lerp = torch.linspace(0, 1.0, num_midpoints + 2, device='cuda').to(x0.dtype)
  return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))

def interp_sheet(G, num_per_sheet, num_midpoints, path, dim_z, device='cuda'):
  # Prepare zs and ys

    zs = interp(torch.randn(num_per_sheet, 1, dim_z, device=device),
                torch.randn(num_per_sheet, 1, dim_z, device=device),
                num_midpoints).view(-1, dim_z)

    # Run the net--note that we've already passed y through G.shared.
    with torch.no_grad():
        out_ims = G(zs).data.cpu()

    image_filename = '{}interp.jpg'.format(path)
    torchvision.utils.save_image(out_ims, image_filename, nrow=num_midpoints + 2, normalize=True)




def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    args = cfg.parse_args()
    
    # Load generator
    gen_net = AutoGAN.models.autogan_cifar10_a.Generator().cuda()

    # Load weights
    checkpoint = torch.load(args.load_path)
    if 'avg_gen_state_dict' in checkpoint:
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        epoch = checkpoint['epoch']
        print('=> loaded checkpoint {} (epoch {})'.format(args.load_path, epoch))
    else:
        gen_net.load_state_dict(checkpoint)
        print('=> loaded checkpoint {}'.format(args.load_path))

    # Sample from generator
    #fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))

    gen_net = gen_net.eval()
    os.makedirs(args.sample_path, exist_ok=True)
    eval_iter = args.num_eval_imgs // args.eval_batch_size

    #interp_sheet(gen_net, 10, 20, '', 128)
    #return

    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                                    torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(args.sample_path, f'iter{iter_idx}_b{img_idx}.png')
            imsave(file_name, img)



if __name__ == "__main__":
    main()
