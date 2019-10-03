import torch
import sys
sys.path.append('./')
import globalvar as gl
import argparse

if __name__ =='__main__':
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

    # batch_size=512
    seed=1234
    # lr=0.01
    use_cuda=True if torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    # m=0.8
    # s=30
    loss_type='cosface'#'baseline','cosface', 'sphereface', 'arcface'
    # latent_dim=2
    # epoch_s=40
    # epoch_e=60
    # latent_dim=2
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

    for m in [0.2,0.3,0.4,0.5,0.6]:
        for s in [10,20,30,40,50]:
            gl.set_value('m',m)
            gl.set_value('s',s)
            import imp
            import train_fMNIST
            imp.reload(train_fMNIST)
            train_fMNIST.main()