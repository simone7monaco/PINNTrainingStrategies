import numpy as np
import torch
from collections import OrderedDict
from utils import choose_optimizer
from tqdm import trange
import wandb

import plotly.express as px


# Multi-layer Perceptron

class LinBlock(torch.nn.Module):
    def __init__(
        self, in_size, out_size, drop_frac=0, act=torch.nn.Tanh,
    ):
        super(LinBlock, self).__init__()
        self.layer = torch.nn.Sequential(torch.nn.Linear(in_size, out_size),
                                         act(),
                                         torch.nn.Dropout(drop_frac))
    def forward(self, x):
        return self.layer(x)
        
        
class DNN(torch.nn.Module):
    def __init__(
        self,
        sizes, drop_frac=0, act=torch.nn.Tanh,
        softmax=False,
        encode_dim=None
    ):
        super(DNN, self).__init__()
        
        self.m = 10
        self.L = encode_dim
        if encode_dim:
            sizes = [2+2*self.m] + sizes[1:] # in_features are now t, 1, cos(wx), sin(wx), ..., cos(mwx), sin(mwx)
            
        layers = []
        self.depth = len(sizes) - 1
        for i in range(len(sizes)-2): 
            layers.append(
                (f'block_{i}', LinBlock(sizes[i], sizes[i+1]))
            )
        layers.append(('output', torch.nn.Linear(sizes[-2], sizes[-1])))
        if softmax == True:
            layers.append(('softmax', torch.nn.Softmax()))

        layerDict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layerDict)
    

    def pbc_encoding(self, x, t):
        w = torch.tensor(2.0 * np.pi / self.L, device=x.device, dtype=torch.float32)
        k = torch.arange(1, self.m + 1, device=x.device)
        H = torch.hstack([t, torch.ones_like(x), torch.cos(k * w * x), torch.sin(k * w * x)])
        
        return H
    
    def forward(self, x):
        if self.L: # Fourier Encoding
            x = self.pbc_encoding(*x)
        elif not type(x) == torch.Tensor:
            x = torch.cat(x, dim=1) # torch.cat([x, t], dim=1)
            
        out = self.layers(x)
        return out
    
    
##############################
# Main Class

class PhysicsInformedNN():
    """PINNs (convection/diffusion/reaction) for periodic boundary conditions."""
    def __init__(self, system, layers, parameters, optimizer_name, lr=1e-3, net='DNN', L=1, act=torch.nn.Tanh,
                 device='cuda' if torch.cuda.is_available() else 'cpu', encode_dim=None
                ):
        self.device = device
        self.system = system

        self.net = net

        # parameters = {nu: <>, beta: <>, ....}
        self.encode_dim = encode_dim
        for k,v in parameters.items():
            setattr(self, k, v)
        self.parameters = [k for k, v in parameters.items() if v !=0]

        self.layers = layers
        if self.net == 'DNN':
            self.dnn = DNN(layers, encode_dim=encode_dim)
        else: # "pretrained" can be included in model path
            self.dnn = torch.load(net).dnn
        
        self.dnn.to(device)
        
        self.L = L
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.optimizer = choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr)
        self.iter = 0
        self.curr_on = None
        self.bar = None
        
        
    def net_u(self, x):
        """The standard DNN that takes (t) | (x,t) --> u.
            If X has more dimension, it must be a list of column tensors (e.g. [x, t])
        """
        u = self.dnn(x)
        return u

    def net_f(self, x, t):
        """ Autograd for calculating the residual for different systems."""
        pass
        
    def boundary_loss(self):
        """ Boundary loss. Not necessary for all problems. """
        return None
    
    def causal_decay_loss(self,x , t):
        r_pred = self.net_f(x, t)
        L_t = (self.T_occurs @ r_pred.pow(2)) / self.T_occurs.sum(1).reshape(-1, 1)
        W = torch.exp(- self.epsilon_weight * (self.M @ L_t)).detach()
        return L_t, W
    
    def loss_pinn(self):
        """ Loss function. """
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        u_pred = self.net_u([self.x_init, self.t_init])
        
        if not self.causal_decay:
            f_pred = self.net_f(self.x_f, self.t_f)
            loss_f = f_pred.pow(2).mean()
        else:
            W, L_t = self.causal_decay_loss(self.x_f, self.t_f)
            if wandb.run: wandb.log({'causal_minweight': W.min().item()}, step=self.iter)            
            loss_f = (W * L_t).mean()

        # initial condition
        loss_u = (self.u - u_pred).pow(2).mean()
        
        loss_b = self.boundary_loss()
        
        loss = loss_u + self.L*loss_f
        if loss_b: loss += loss_b

        if self.pbar is not None:
            self.pbar.set_postfix({'Loss': loss.item()})
            
        if wandb.run:
            wandb.log({"bound_loss": loss_b}, step=self.iter)
            wandb.log({"fun_loss": loss_f}, step=self.iter)
            wandb.log({"init_loss": loss_u}, step=self.iter)
            wandb.log({"loss": loss}, step=self.iter)
        
        if loss.requires_grad:
            loss.backward()
        
        self.iter += 1
        if wandb.run: wandb.log({'epoch': self.iter})
    
        return loss

    def train(self, epochs=3000, curriculum:dict=None, Dom=None, U_real=None):
        self.epochs = epochs
        self.dnn.train()
        
        if curriculum is not None:
            self.curriculum_learning(curriculum)

            for curr_val in self.curr_range:
                setattr(self, self.curr_on, curr_val)
                
                # Longer training on last step
                if curr_val == self.curr_range[-1] and curriculum['version'] == 2: 
                    self.curr_epochs = self.epochs - self.iter
                if wandb.run: wandb.log({self.curr_on: curr_val}, step=self.iter)
                self.pbar = trange(self.curr_epochs, desc = f'[CURR] Training on {self.curr_on}: {curr_val:.3}')
                for _ in self.pbar:
                    self.dnn.train()
                    self.optimizer.step(self.loss_pinn)
                    if Dom is not None and self.iter % min(self.epochs//10, 100) == 0 and wandb.run is not None:
                        self.dnn.eval()
                        self.validate(Dom, U_real)
                    
        else:
            self.pbar = trange(epochs, desc = f'[{self.net}{"/causal" if self.causal_decay else ""}{"/encPBC" if self.encode_dim else ""}] training')
            for _ in self.pbar:
                self.dnn.train()
                self.optimizer.step(self.loss_pinn)
                
                if Dom is not None and self.iter % 100 == 0 and wandb.run is not None:
                        self.dnn.eval()
                        self.validate(Dom, U_real)
        return
            
    def curriculum_learning(self, curriculum):
        print(f"Using curriculum learning on param {curriculum['curr_on']} with target {getattr(self, curriculum['curr_on'])}\n")
        self.curr_on = curriculum['curr_on']
        self.curr_range = np.linspace(*sorted([curriculum['curr_initval'], getattr(self, self.curr_on)]),
                                      curriculum['curr_steps'])
        if curriculum['curr_initval'] > getattr(self, self.curr_on):
            self.curr_range = self.curr_range[::-1] # the bigger the easier
        setattr(self, self.curr_on, self.curr_range[0])
        
        if curriculum['version'] == 1:
            self.curr_epochs = self.epochs // curriculum['curr_steps']
        elif curriculum['version'] == 2:
            assert curriculum['curr_epochs'] * curriculum['curr_steps'] < self.epochs, "More epochs are needed for Curriculum v2"
            self.curr_epochs = curriculum['curr_epochs']
        else:
            raise ValueError(f"Wrong version selection (selected {curriculum['version']} instead of 1 or 2)")
        
    def predict(self, X):
        x = torch.tensor(X[:, 0:1]).float().to(self.device)
        t = torch.tensor(X[:, 1:2]).float().to(self.device)

        self.dnn.eval()
        u = self.net_u([x, t])
        u = u.detach().cpu().numpy()
        return u 
    
    
class PINN_pbc(PhysicsInformedNN):
    def __init__(self, system, Dom_init, U_init, Dom_train, Dom_lbound, Dom_ubound, 
                 layers, parameters, optimizer_name, lr=1e-3, net='DNN', L=1, act=torch.nn.Tanh,
                 device='cuda' if torch.cuda.is_available() else 'cpu', causal_decay=False, 
                 epsilon_weight=None, encode_dim=None):
        
        super(PINN_pbc, self).__init__(system=system, layers=layers, parameters=parameters,
                                       optimizer_name=optimizer_name, lr=lr, net=net, L=L, act=act,
                                       device=device, encode_dim=encode_dim)
        
        self.encode_dim = encode_dim

        self.x = Dom_init[:, 0]
        self.t = Dom_lbound[:, 1]
        
        self.x_init = torch.tensor(Dom_init[:, 0:1], requires_grad=True).float().to(device)
        self.t_init = torch.tensor(Dom_init[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(U_init, requires_grad=True).float().to(device)
        
        self.x_f = torch.tensor(Dom_train[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(Dom_train[:, 1:2], requires_grad=True).float().to(device)
        
        self.x_lbound = torch.tensor(Dom_lbound[:, 0:1], requires_grad=True).float().to(device)
        self.t_lbound = torch.tensor(Dom_lbound[:, 1:2], requires_grad=True).float().to(device)
        self.x_ubound = torch.tensor(Dom_ubound[:, 0:1], requires_grad=True).float().to(device)
        self.t_ubound = torch.tensor(Dom_ubound[:, 1:2], requires_grad=True).float().to(device)
        
        G = np.full(Dom_train.shape[0], 0.) # Source for convection: till now it is hardcoded
        self.G = torch.tensor(G, requires_grad=True).reshape(-1, 1).float().to(device)
        
        # Causal training
        self.causal_decay = causal_decay
        self.T_occurs = torch.stack([t == self.t_f.unique() for t in self.t_f]).float().to(self.device).T
        self.M = torch.triu(torch.ones((self.T_occurs.shape[0], self.T_occurs.shape[0])), diagonal=1).to(self.device).T
        self.epsilon_weight = epsilon_weight
        
    def net_f(self, x, t):
        """ Autograd for calculating the residual for different systems."""
        
        u = self.net_u([x, t])

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
            )[0]

        # f = get_residuals(self.system)
        if self.system == 'convection':
            f = u_t + self.beta*u_x - self.G
        elif self.system == 'burger':
            f = u_t + u*u_x - self.nu*u_xx
        elif self.system == 'allencahn':
            f = u_t + 5 * u**3 - 5 * u - self.nu * u_xx
        elif self.system == 'reaction-diffusion':
            f = u_t - self.nu * u_xx - self.rho * u * (1 - u)
        elif self.system == 'ks':
            u_xxx = torch.autograd.grad(
                u_xx, x,
                grad_outputs=torch.ones_like(u_xx),
                retain_graph=True,
                create_graph=True
                )[0]
            u_xxxx = torch.autograd.grad(
                u_xxx, x,
                grad_outputs=torch.ones_like(u_xxx),
                retain_graph=True,
                create_graph=True
                )[0]
            f = u_t + self.alpha * u * u_x + self.beta * u_xx + self.gamma * u_xxxx
        else:
            raise NotImplementedError("System not valid.")
        return f
    
    def net_b_derivatives(self, u_lb, u_ub, x_lbound, x_ubound):
        """For taking BC derivatives."""

        u_lb_x = torch.autograd.grad(
            u_lb, x_lbound,
            grad_outputs=torch.ones_like(u_lb),
            retain_graph=True,
            create_graph=True
            )[0]

        u_ub_x = torch.autograd.grad(
            u_ub, x_ubound,
            grad_outputs=torch.ones_like(u_ub),
            retain_graph=True,
            create_graph=True
            )[0]

        return u_lb_x, u_ub_x
    
    def boundary_loss(self):
        """Boundary loss for PBC"""
        if self.encode_dim: return None
        
        u_pred_lb = self.net_u([self.x_lbound, self.t_lbound])
        u_pred_ub = self.net_u([self.x_ubound, self.t_ubound])
        
        loss_b = (u_pred_lb - u_pred_ub).pow(2).mean()
        
        if self.system != 'convection': # first order PDE
            u_pred_lb_x, u_pred_ub_x = self.net_b_derivatives(u_pred_lb, u_pred_ub, 
                                                              self.x_lbound, self.x_ubound)
            loss_b += (u_pred_lb_x - u_pred_ub_x).pow(2).mean()
        return loss_b
    
    def validate(self, Dom, U_real, is_last=False):
        U_pred = self.predict(Dom).reshape(self.t.size, 
                                            self.x.size)
        mse = np.power(U_pred-U_real, 2).mean()
        wandb.log({'valid_mse': mse}, step=self.iter)
        
        if is_last:
            fig = px.imshow(U_pred.T, x=self.t, y=self.x, aspect='auto', origin='lower', 
                            labels={'x': 'T', 'y': 'X'},
                            color_continuous_scale=px.colors.sequential.Turbo,
                            title=f"Epoch {self.iter} (mse: {mse:.3})"
                           )
            wandb.log({"Evaluation": fig}, step=self.iter)