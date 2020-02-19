''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import functools
import math
import numpy as np
from tqdm import tqdm, trange

import os
from collections import defaultdict


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import utils

import pdb

def run(config):
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}
                
  # Optionally, get the configuration from the state dict. This allows for
  # recovery of the config provided only a state dict and experiment name,
  # and can be convenient for writing less verbose sample shell scripts.
  if config['config_from_name']:
    utils.load_weights(None, None, state_dict, config['weights_root'], 
                       config['experiment_name'], config['load_weights'], None,
                       strict=False, load_optim=False)
    # Ignore items which we might want to overwrite from the command line
    for item in state_dict['config']:
      if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
        config[item] = state_dict['config'][item]
  
  # update config (see train.py for explanation)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  config = utils.update_config_roots(config)
  config['skip_init'] = True
  config['no_optim'] = True
  device = 'cuda'
  
   
  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True
  
  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)
  
  G = model.Generator(**config).cuda()
   
     
  # Load weights
  print('Loading weights...')
  # Here is where we deal with the ema--load ema weights or load normal weights
  utils.load_weights(G if not (config['use_ema']) else None, None, state_dict, 
                     config['weights_root'], experiment_name, config['load_weights'],
                     G if config['ema'] and config['use_ema'] else None,
                     strict=False, load_optim=False)

  # Update batch size setting used for G
  G_batch_size = max(config['G_batch_size'], config['batch_size']) 
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'], 
                             z_var=config['z_var'])
  
  if config['G_eval_mode']:
    print('Putting G in eval mode..')
    G.eval()
  else:
    print('G is in %s mode...' % ('training' if G.training else 'eval'))
  
  sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)  
  brief_expt_name = config['experiment_name'][-30:]  
    
    
  # Prepare sample sheets
  if config['sample_sheets']:
    print('Preparing conditional sample sheets...')
    folder_number=config['sample_sheet_folder_num']
    if folder_number == -1:
      folder_number = config['load_weights']
    utils.sample_sheet(G, classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']], 
                         num_classes=config['n_classes'], 
                         samples_per_class=10, parallel=config['parallel'],
                         samples_root=config['samples_root'], 
                         experiment_name=experiment_name,
                         folder_number=folder_number,
                         z_=z_,)
  # Sample interp sheets
  if config['sample_interps']:
    print('Preparing interp sheets...')
    folder_number=config['sample_sheet_folder_num']
    if folder_number == -1:
      folder_number = config['load_weights']
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
      utils.interp_sheet(G, num_per_sheet=16, num_midpoints=8,
                         num_classes=config['n_classes'], 
                         parallel=config['parallel'], 
                         samples_root=config['samples_root'], 
                         experiment_name=experiment_name,
                         folder_number=int(folder_number), 
                         sheet_number=0,
                         fix_z=fix_z, fix_y=fix_y, device='cuda')
  # Sample random sheet
  if config['sample_random']:
    print('Preparing random sample sheet...')
    images, labels = sample()    
    torchvision.utils.save_image(images.float(),
                                 '%s/%s/%s.jpg' % (config['samples_root'], experiment_name, config['load_weights']),
                                 nrow=int(G_batch_size**0.5),
                                 normalize=True)

  
  
  
      
def main():
  # parse command line and run    
  parser = utils.prepare_parser()
  parser = utils.add_sample_parser(parser)
  config = vars(parser.parse_args())
  print(config)
  if config['sample_multiple']:
    suffixes = config['load_weights'].split(',')
    for suffix in suffixes:
      config['load_weights'] = suffix
      run(config)
  else:
    run(config)
  
if __name__ == '__main__':    
  main()
