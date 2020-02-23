import torch
from torch.autograd import Variable
import torch.nn as nn
import time
import numpy as np

from utils import *
from gan.sample import sample_gan

def adjust_learning_rate(optimizer, epoch, lr, steps, decay):
    # Decay lr every n_epochs 
    for step in steps:
        if epoch >= step:
            lr *= decay
        else:
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    '''
    Computes the precision@k for the specified values of k
    From: https://github.com/vikasverma1077/manifold_mixup/blob/5485529c02acc1f10a2ba2a32d99303b6e1b6034/supervised/main.py
    '''

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(net, train_loader, test_loader, optimizer, criterion, generator, history, device, config):
    '''
    Main training loop
    '''
    if not history:
        history = RecorderMeter(config['epochs'])
        best_acc = 0
    else:
        history.expand(config['epochs'])
        best_acc = history.max_accuracy(isTrain=False)
        print_and_log('Checkpoint\'s best validation accuracy: {}'.format(best_acc), config['log'])

    start_time = time.time()
    print_and_log('\nStarting to train...', config['log'])

    for epoch in range(config['start_epoch'], config['epochs']+1):
        epoch_start = time.time()

        if config['schedule'] == 'step':
            adjust_learning_rate(optimizer, epoch, config['lr'], config['steps'], config['step_size'])    
        print_and_log('\nEpoch [{}/{}] Learning Rate: {}'.format(epoch, config['epochs'], optimizer.param_groups[0]['lr']), config['log'])

        # Train for an epoch
        train_acc, train_loss = train_epoch(net, train_loader, optimizer, criterion, generator, epoch, device, config)

        # Validate model
        val_acc, val_loss = validate(net, test_loader, criterion, device, config)

        # Save acc and losses
        history.update(epoch, train_loss, train_acc, val_loss, val_acc)
        history.plot_curve(config['output_path']+'plot.png')

        # Save checkpoint
        is_best = False
        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc

        save_checkpoint({
            'epoch': epoch,
            'arch': config['arch'],
            'state_dict': net.state_dict(),
            'history': history,
            'optimizer': optimizer.state_dict(),
        }, config['output_path'], is_best)

        # Print epoch stats
        hours, minutes, seconds = calculate_time(epoch_start, time.time())
        print_and_log("Epoch {} Completed in {}h {}m {:04.2f}s"
              .format(epoch, hours, minutes, seconds), config['log'])

    # Print training time
    hours, minutes, seconds = calculate_time(start_time, time.time())
    print_and_log('\nTraining completed in {}h {}m {:04.2f}s'.format(hours, minutes, seconds), config['log'])
    print_and_log('Best validation accuracy: {} ({})'.format(best_acc, 100-best_acc), config['log'])


def train_epoch(net, train_loader, optimizer, criterion, generator, epoch, device, config):
    ''' Train for 1 epoch '''

    net.train()

    losses = AverageMeter()
    accs = AverageMeter()

    for i, (inp, target) in enumerate(train_loader):
        if config['use_gan'] and i%config['use_gan_freq'] == 0:
            inp, target = sample_gan(generator, config['batch_size'], config['truncation'], device)
        else:
            target = target.long()
            inp, target = inp.to(device), target.to(device)


        # Forward pass
        if config['training'] == 'vanilla':
            output, reweighted_targets = net(inp, target, smoothing=config['smoothing'], device=device)
            loss = criterion(output, reweighted_targets.to(device))

        elif config['training'] == 'mixup':
            if np.random.rand(1) < config['mix_prob']:
                output, reweighted_targets = net(inp, target, mixup=True, mixup_alpha=config['mixup_alpha'], 
                                                 smoothing=config['smoothing'], device=device)
                loss = criterion(output, reweighted_targets.to(device))
            else:
                output, reweighted_targets = net(inp, target, smoothing=config['smoothing'], device=device)
                loss = criterion(output, reweighted_targets.to(device))

        elif config['training'] == 'manifold_mixup':
            if np.random.rand(1) < config['mix_prob']:
                output, reweighted_targets = net(inp, target, mixup_hidden=True, mixup_alpha=config['mixup_alpha'], 
                                                 smoothing=config['smoothing'], device=device)
                loss = criterion(output, reweighted_targets.to(device))
            else:
                output, reweighted_targets = net(inp, target, smoothing=config['smoothing'], device=device)
                loss = criterion(output, reweighted_targets.to(device))

        elif config['training'] == 'cutmix':
            if np.random.rand(1) < config['mix_prob']:
                output, reweighted_targets = net(inp, target, cutmix=True, mixup_alpha=config['mixup_alpha'], 
                                                 smoothing=config['smoothing'], device=device)
                loss = criterion(output, reweighted_targets.to(device))
            else:
                output, reweighted_targets = net(inp, target, smoothing=config['smoothing'], device=device)
                loss = criterion(output, reweighted_targets.to(device))

        elif config['training'] == 'manifold_cutmix':
            if np.random.rand(1) < config['mix_prob']:
                output, reweighted_targets = net(inp, target, cutmix_hidden=True, mixup_alpha=config['mixup_alpha'], 
                                                 smoothing=config['smoothing'], device=device)
                loss = criterion(output, reweighted_targets.to(device))
            else:
                output, reweighted_targets = net(inp, target, smoothing=config['smoothing'], device=device)
                loss = criterion(output, reweighted_targets.to(device))
        
        else:
            raise NotImplementedError('{} is not an available training method'.format(config['training']))


        # Save loss and acc
        [acc] = accuracy(output, target)
        losses.update(loss.item(), inp.size(0))
        accs.update(acc.item(), inp.size(0))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%config['batch_log_rate'] == 0:
            print_and_log('Epoch [{}/{}], Batch [{}/{}] Loss: {} Acc: {}'.format(epoch, config['epochs'], i+1, len(train_loader), 
                                                                                 losses.avg, accs.avg), config['log'])
        
    print_and_log('Epoch [{}/{}] Training Loss: {} Acc: {}'.format(epoch, config['epochs'], losses.avg, accs.avg), config['log'])
    return accs.avg, losses.avg

def validate(net, val_loader, criterion, device, config):
    net.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        for i, (inp, target) in enumerate(val_loader):
            inp, target = inp.to(device), target.to(device)

            # Forward pass
            output, reweighted_targets = net(inp, target, device=device)
            loss = criterion(output, reweighted_targets.to(device))

            # Save loss and acc
            [acc] = accuracy(output.data, target)
            losses.update(loss.item(), inp.size(0))
            accs.update(acc.item(), inp.size(0))

    print_and_log('Validation Loss: {} Acc: {}'.format(losses.avg, accs.avg), config['log'])
    return accs.avg, losses.avg