import numpy as np
import yaml
import os
import random
import torch
from utils import *
from solvers import *
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
from DNN import PINN_pbc
import argparse
import wandb
from pathlib import Path
from scipy import io as spio
import plotly.express as px



def get_args():
    parser = argparse.ArgumentParser(description='PINN launcher')
    parser.add_argument('--system', type=str, default='convection')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--N_f', type=int, default=1000)
    parser.add_argument('--optimizer_name', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--L', type=float, default= 1.0, help='lambda')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--layers', type=str, default= '[50, 50, 50, 50, 1]')
    parser.add_argument('--net', type=str, default= 'DNN')
    parser.add_argument('--u0_str', type=str, default=None)
    parser.add_argument('--nx', type=int, default=None)
    parser.add_argument('--nt', type=int, default=None)
    parser.add_argument('--parameters', type=str, default=None)
    parser.add_argument('--curriculum', type=str, default=None, help='If specified, performs curriculum learning algorithm on the argument parameter.')
    parser.add_argument('--curr_steps', type=int, default=None)
    parser.add_argument('--curr_epochs', type=int, default=None, help='Epochs if using Curriculum learning') # TODO: fix con epochs
    parser.add_argument('--encode_inputs', nargs='?', default=False, const=True, type=str2bool, help="Encode NN inputs to enforce PBC.")
    parser.add_argument('--training', type=str, default='Vanilla', help='Can be either "causal", "curriculum", "curriculumv1", or None (i.e. vanilla)')
    parser.add_argument('--epsilon_weight', type=float, default=None, help="Temperature parameter for the causal weights")
    parser.add_argument('--no_wandb', nargs='?', default=False, const=True, type=str2bool, help="Prevent Wandb to log the run.")
    parser.add_argument('--save_outputs', nargs='?', default=False, const=True, type=str2bool, help="Save the final outputs (mse curve, loss and final prediction).")
    parser.add_argument('--save_model', nargs='?', default=False, const=True, type=str2bool, help="Save the model.")
    args = parser.parse_args()
    
    if args.layers is not None:
        args.layers = eval(args.layers)
    if args.parameters is not None:
        args.parameters = eval('{'+args.parameters+'}')
    return args

def main(args, device):
    log_path = Path(f"logs/{args.system}")
    log_path.mkdir(exist_ok=True, parents=True)
    
    with open("configs.yaml", 'r') as c:
        configs = yaml.load(c, Loader=yaml.SafeLoader)
    
    for k, v in configs[args.system].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    if 'curriculum' not in args.training: 
        for k in ['curriculum', 'curr_initval', 'curr_steps', 'curr_epochs']:
            setattr(args, k, None)
    if args.training != 'causal': args.epsilon_weight = None
    
    set_seed(args.seed)

    while args.N_f > args.nx * args.nt:
        args.nx *= 2
        args.nt *= 2
    print(f"\nSystem to solve: {args.name} ({args.nx}×{args.nt})\n")
    if args.system == 'convection':
        x = np.linspace(0, 2*np.pi, args.nx, endpoint=False).reshape(-1, 1) # not inclusive
        t = np.linspace(0, 1, args.nt).reshape(-1, 1)
        U_field = convection_diffusion(x.ravel(), t.ravel(), args.u0_str,
                                      nu=0, rho=0, beta=args.parameters['beta'],
                                      nx=args.nx, nt=args.nt).reshape(t.size, x.size)
    elif args.system in ['reaction-diffusion', 'allencahn']:
        x = np.linspace(*args.xrange, args.nx, endpoint=False).reshape(-1, 1) # not inclusive
        t = np.linspace(0, 1, args.nt).reshape(-1, 1)
        U_field = fft2_discrete_sol(x.ravel(), t.ravel(), args.u0_str, system=args.system,
                                    **args.parameters)
    else:
        raise NotImplementedError("System not valid.")
        
        
    X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
    Dom = np.hstack((X.reshape(-1,1), T.reshape(-1,1))) # all the x, t "test" data

    x_inner = x[1:]
    t_inner = t[1:]
    X_inner, T_inner = np.meshgrid(x_inner, t_inner)
    Dom_inner = np.hstack((X_inner.reshape(-1,1), T_inner.reshape(-1,1)))

    Dom_train = sample_random(Dom_inner, args.N_f) # training dataset
      
    Dom_init = np.hstack((X[0,:].reshape(-1, 1), T[0,:].reshape(-1, 1))) # ([-x_end, +x_end],0)
    U_init = U_field[0:1, :].T # u([-x_end, +x_end],0)

    Dom_lbound = np.hstack((X[:,0].reshape(-1, 1), T[:,0].reshape(-1, 1))) # (x_min,[t])
    U_lbound = U_field[:, 0:1]

    # BC at x=2pi
    Dom_ubound = np.hstack((np.full_like(t, X[:,-1].reshape(-1, 1) + x[1] - x[0]), 
                            T[:,-1].reshape(-1, 1))
                          ) # (x_max,[t]), considering pbc
    U_ubound = U_field[:, -1].reshape(-1, 1)

    if args.layers[0] != Dom_train.shape[-1]:
        args.layers.insert(0, Dom_train.shape[-1])
    
    tags = []
    if args.training: tags.append(args.training)
    if not tags: tags.append('Vanilla')
    
    causal_decay = True if args.training == 'causal' else False

    if not args.no_wandb:
        wandb.init(config=args, project="convection-pinn", tags=tags)
    exp_tit = f'{"_".join(tags)}_k_{args.seed}'
        
    enc_dim = np.diff(args.xrange) if args.encode_inputs else None 
    
    model = PINN_pbc(args.system, Dom_init, U_init, Dom_train, Dom_lbound, Dom_ubound, 
                     args.layers, args.parameters, args.optimizer_name, args.lr, 
                     args.net, args.L, act=torch.nn.Tanh, device=device, 
                     causal_decay=causal_decay, epsilon_weight=args.epsilon_weight,
                     encode_dim=enc_dim)

    curr_params = {'curr_on': args.curriculum, 
                   'curr_steps': args.curr_steps,
                   'curr_epochs': args.curr_epochs,
                   'curr_initval': args.curr_initval,
                   'version': 1 if 'v1' in args.training else 2,
                  }
    if 'curriculum' not in args.training: curr_params = None
    
    model.train(args.epochs, curriculum=curr_params,
                Dom=Dom, U_real=U_field)
    
    if wandb.run is not None:
        if args.curriculum: wandb.log({model.curr_on: getattr(model, model.curr_on)}, step=model.iter)
        model.validate(Dom, U_real=U_field, is_last=True)
        print("epochs:", model.iter)
        wandb.finish()
        
    if args.save_outputs:
        with open(log_path / f'{exp_tit}_result.npy', 'wb') as f:
            np.save(f, model.predict(Dom))
        
    if args.save_model:
        torch.save(model.state_dict(), log_path / f"{exp_tit}.pt")

    return


if __name__ == '__main__':
    '''
    convection: 𝑢𝑡−𝜈⋅𝑢𝑥𝑥−𝜌𝑢(1−𝑢)=0
    '''
    device = torch.device(f'cuda:{free_cuda_id()}') if torch.cuda.is_available() else 'cpu'
    args = get_args()
    main(args, device)
    
