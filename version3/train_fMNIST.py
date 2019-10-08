import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import globalvar as gl
import pdb

from models import ConvAngularPen, ConvBaseline
from tensorboardX import SummaryWriter


#######################################################################################
########################################加载参数########################################
batch_size=gl.get_value('batch_size')
seed=gl.get_value('seed')
lr=gl.get_value('lr')
use_cuda=gl.get_value('use_cuda')
device=gl.get_value('device')
m=gl.get_value('m')
s=gl.get_value('s')
loss_type=gl.get_value('loss_type')
latent_dim=gl.get_value('latent_dim')
epoch_s=gl.get_value('epoch_s')
epoch_e=gl.get_value('epoch_e')
latent_dim=gl.get_value('latent_dim')
logdir=gl.get_value('logdir')

def main():

    writer = SummaryWriter(logdir)
    torch.manual_seed(seed)



    train_ds = datasets.FashionMNIST(
                                    root = './data',
                                    train=True,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Normalize(mean=(0.1307,), std=(0.3081,))]),
                                    download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_ds,
                                            batch_size=batch_size,
                                            shuffle=True)
    example_loader = torch.utils.data.DataLoader(dataset=train_ds,
                                            batch_size=10000,
                                            shuffle=False)


    os.makedirs('{}/result'.format(logdir), exist_ok=True)

    print('Training {} model....'.format(loss_type))
    model_am = train_am(train_loader,example_loader,writer)


def train_am(train_loader,example_loader,writer):
    model = ConvAngularPen(loss_type=loss_type).to(device)
    path='{}/result/{}_epoch_{}_m{}_s{}.pkl'.format(logdir,loss_type,epoch_s,m,s)
    if epoch_s>0:
        model.load_state_dict(torch.load(path))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_s):
        if((epoch+1) % 8 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/4
    global_step=epoch_s*len(train_loader)
    for epoch in tqdm(range(epoch_s,epoch_e)):
        for i, (feats, labels) in enumerate(train_loader):
            feats = feats.to(device)
            labels = labels.to(device)
            loss = model(feats, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step=global_step+1
            writer.add_scalars('{}'.format(loss_type), {'train L': loss.item()}, global_step=global_step)
        if((epoch+1) % 8 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/4


        for i, (feats, labels) in enumerate(example_loader):
            with torch.no_grad():
                feats = feats[:200].to(device)
                labels = labels[:200].to(device)
                loss = model(feats, labels=labels)
                writer.add_scalars('{}'.format(loss_type), {'test L': loss.item()}, global_step=global_step)

        writer.add_scalars('{}'.format(loss_type), {'epoch': 1}, global_step=epoch)
        if (epoch+1)%10==0:
            torch.save(model.state_dict(), '{}/result/{}_epoch_{}_m{}_s{}.pkl'.format(logdir,loss_type,epoch+1,m,s))
    return model.cpu()

def get_embeds(model, example_loader):
    loader=example_loader
    device=gl.get_value('device')
    model = model.to(device).eval()
    full_embeds = []
    full_labels = []
    with torch.no_grad():
        for i, (feats, labels) in enumerate(loader):
            feats = feats[:100].to(device)
            full_labels.append(labels[:100].cpu().detach().numpy())
            embeds = model(feats, embed=True)
            full_embeds.append(embeds.detach().cpu().numpy())
    return np.concatenate(full_embeds), np.concatenate(full_labels)