import torch
from argparse import ArgumentParser
import yaml
import pprint
import torchvision
import torch.backends.cudnn as cudnn

from BigGANmh import Generator

def load_config():
    ''' Load config file from command line '''

    # Get the config path 
    parser = ArgumentParser()
    parser.add_argument('-c', '--config',
                        dest='config_path',
                        help='config file path',
                        required=True)
    
    path = parser.parse_args().config_path
    # Load the config file
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Print the config file contents
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    return config

def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda', z_var=1.0):
  z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
  z_.init_distribution('normal', mean=0, var=z_var)
  z_ = z_.to(device, torch.float32)   
  
  y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
  y_.init_distribution('categorical',num_categories=nclasses)
  y_ = y_.to(device, torch.int64)
  return z_, y_

class Distribution(torch.Tensor):
  # Init the params of the distribution
  def init_distribution(self, dist_type, **kwargs):    
    self.dist_type = dist_type
    self.dist_kwargs = kwargs
    if self.dist_type == 'normal':
      self.mean, self.var = kwargs['mean'], kwargs['var']
    elif self.dist_type == 'categorical':
      self.num_categories = kwargs['num_categories']

  def sample_(self):
    if self.dist_type == 'normal':
      self.normal_(self.mean, self.var)
    elif self.dist_type == 'categorical':
      self.random_(0, self.num_categories)    
    # return self.variable
    
  # Silly hack: overwrite the to() method to wrap the new object
  # in a distribution as well
  def to(self, *args, **kwargs):
    new_obj = Distribution(self)
    new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
    new_obj.data = super().to(*args, **kwargs)    
    return new_obj

def sample(G, z, y):
  with torch.no_grad():
    z.sample_()
    y.sample_()

    G_z = G(z, G.shared(y))
    return G_z, y



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True

config = load_config()

gen = Generator().to(device)
gen.load_state_dict(torch.load(config['gan_weights']))
gen.eval()

z, y = prepare_z_y(config['batch_size'], dim_z=128, nclasses=10, device=device,)

images, labels = sample(gen, z, y)    
print(torch.min(images))
print(torch.max(images))

torchvision.utils.save_image(images.float(),
                             'samples/samples.jpg',
                             nrow=int(config['batch_size']**0.5),
                             normalize=True)


