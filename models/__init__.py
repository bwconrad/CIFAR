from torch.nn import init
from models.preresnet import preactresnet18, preactresnet34, preactresnet50, preactresnet101, preactresnet152
from utils import print_and_log

def load_model(config):
    arch_name = config['arch']
    print_and_log('Setting up {} model...'.format(arch_name), config['log'])
    
    if arch_name == 'preactresnet18':
        net = preactresnet18(n_classes=config['n_classes'], initial_channels=config['initial_channels'], mixup_layers=config['mixup_layers'])
        init_weights(net, config, config['weight_init'], config['weight_init_gain'])
        return net
    elif arch_name == 'preactresnet34':
        net = preactresnet34(n_classes=config['n_classes'], initial_channels=config['initial_channels'], mixup_layers=config['mixup_layers'])
        init_weights(net, config, config['weight_init'], config['weight_init_gain'])
        return net
    elif arch_name == 'preactresnet50':
        net = preactresnet50(n_classes=config['n_classes'], initial_channels=config['initial_channels'], mixup_layers=config['mixup_layers'])
        init_weights(net, config, config['weight_init'], config['weight_init_gain'])
        return net
    elif arch_name == 'preactresnet101':
        net = preactresnet101(n_classes=config['n_classes'], initial_channels=config['initial_channels'], mixup_layers=config['mixup_layers'])
        init_weights(net, config, config['weight_init'], config['weight_init_gain'])
        return net
    elif arch_name == 'preactresnet152':
        net = preactresnet152(n_classes=config['n_classes'], initial_channels=config['initial_channels'], mixup_layers=config['mixup_layers'])
        init_weights(net, config, config['weight_init'], config['weight_init_gain'])
        return net
    else:
        raise NotImplementedError('{} is not an available architecture'.format(arch_name))

def init_weights(net, config, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__

        # Conv and lin layers
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method {} is not implemented'.format(init_type))

            # Biases
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        # Batch norms
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)


    print_and_log('Initializing weights as {}'.format(init_type.upper()), config['log'])
    net.apply(init_func)