import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import argparse
from torchvision import datasets, transforms
import torch
import globalvar as gl

train_ds = datasets.FashionMNIST(
                                    root = './data',
                                    train=True,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Normalize(mean=(0.1307,), std=(0.3081,))]),
                                    download=False)
example_loader = torch.utils.data.DataLoader(dataset=train_ds,
                                            batch_size=10000,
                                            shuffle=True)

def initparam():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',default=512,type=int)
    parser.add_argument('--lr',default=0.01,type=float)
    parser.add_argument('--m',default=0.4,type=float)
    parser.add_argument('--s',default=30,type=float)
    parser.add_argument('--latent_dim',default=2,type=int)
    parser.add_argument('--epoch_s',default=0,type=int)
    parser.add_argument('--epoch_e',default=40,type=int)
    args=parser.parse_args()

    batch_size=args.batch_size
    lr=args.lr
    m=args.m
    s=args.s
    latent_dim=args.latent_dim
    epoch_s=args.epoch_s
    epoch_e=args.epoch_e    

    seed=1234
    use_cuda=True if torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")

    loss_type='cosface'
    logdir='./log{}d_m{}_s{}'.format(latent_dim,m,s)

    gl._init()
    gl.set_value('batch_size',batch_size)
    gl.set_value('seed',seed)
    gl.set_value('lr',lr)
    gl.set_value('use_cuda',use_cuda)
    gl.set_value('device',device)
    gl.set_value('m',m)
    gl.set_value('s',s)
    gl.set_value('loss_type',loss_type)
    gl.set_value('epoch_s',epoch_s)
    gl.set_value('epoch_e',epoch_e)
    gl.set_value('latent_dim',latent_dim)
    gl.set_value('logdir',logdir)

if __name__ =='__main__':
    initparam()

    from models import ConvAngularPen
    from train_fMNIST import get_embeds
    epoch=80
    gl.set_value('m',0.4)
    gl.set_value('s',30)

    logdir='./v2/log{}d_m{}_s{}'.format(gl.get_value('latent_dim'),gl.get_value('m'),gl.get_value('s'))
    gl.set_value('logdir',logdir)

    model = ConvAngularPen(loss_type='arcface').to(gl.get_value('device'))
    path='{}/result/{}_epoch_{}_m{}_s{}.pkl'.format(
        gl.get_value('logdir'),
        gl.get_value('loss_type'),
        epoch,
        gl.get_value('m'),
        gl.get_value('s'))
    model.load_state_dict(torch.load(path))
    embeds,labels=get_embeds(model, example_loader)
    savedata=pd.DataFrame(np.hstack((embeds,labels.reshape(-1,1))))

    savedata.to_csv('{}/result/raw_feature_{}d_m{}_s{}.csv'.format(logdir,
    gl.get_value('latent_dim'),
    gl.get_value('m'),
    gl.get_value('s')))