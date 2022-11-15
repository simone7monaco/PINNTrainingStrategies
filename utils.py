import torch
import numpy as np
import pandas as pd
import argparse
import subprocess

from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def sample_random(S, N):
    """Given an array of (x,t) points, sample N points from this."""

    idx = np.random.choice(S.shape[0], N, replace=False)
    idx.sort()
    S_sampled = S[idx, :]
    assert (np.diff(idx) > 0).all()
    return S_sampled


def function(u0: str):
    """Initial condition, string --> function."""

    funs = {
        'sin(x)': lambda x: np.sin(x),
        '-sin(x)': lambda x: -np.sin(x),
        '-sin(pix)': lambda x: -np.sin(np.pi*x),
        '-sin(2pix)': lambda x: -np.sin(2*np.pi*x),
        'sin(pix)': lambda x: np.sin(np.pi*x),
        'sin^2(x)': lambda x: np.sin(x)**2,
        'sin(x)cos(x)': lambda x: np.sin(x)*np.cos(x),
        'x^2*cos(pix)': lambda x: x**2 * np.cos(np.pi*x),
        '0.1sin(x)': lambda x: 0.1*np.sin(x),
        '0.5sin(x)': lambda x: 0.5*np.sin(x),
        '10sin(x)': lambda x: 10*np.sin(x),
        '50sin(x)': lambda x: 50*np.sin(x),
        '1+sin(x)': lambda x: 1 + np.sin(x),
        '2+sin(x)': lambda x: 2 + np.sin(x),
        '6+sin(x)': lambda x: 6 + np.sin(x),
        '10+sin(x)': lambda x: 10 + np.sin(x),
        'sin(2x)': lambda x: np.sin(2*x),
        'tanh(x)': lambda x: np.tanh(x),
        '2x': lambda x: 2*x,
        'x^2': lambda x: x**2,
        'gauss':  lambda x: np.exp(-np.power(x - np.pi, 2.) / (2 * np.power(np.pi/4, 2.)))
    }
    return funs[u0]

def data_loss(net, x, t, y):
    y_ = net((x, t))
    
    return torch.nn.functional.mse_loss(y.float(), y_)



def downsampling(Spaces, y=None, step=1, kind:str='uniform', create_dl=True, hparams=None):
    mask = pd.DataFrame(np.hstack([Spaces, 2*np.ones(Spaces.shape[0]).reshape(-1,1)]))
    
    val = mask.columns[-1]
    d = Spaces.shape[1]
    if kind == 'uniform':
        mask.iloc[np.random.choice(mask.index, Spaces.shape[0]//step, replace=False), -1] = 0
        # mask.iloc[np.all([mask[c].isin(mask[c].unique()[::step])\
        #                   for c in mask.columns[:-1]],
        #                  axis=0), -1] = 1
    elif kind == 'portion':
        for c in mask.columns[:-1]: 
            mask[c] -= mask[c].min()
            mask[c] /= mask[c].max()
        mask.iloc[np.all([mask[c] < 1 /step**(1/d)\
                          for c in mask.columns[:-1]],
                         axis=0), -1] = 0
    elif kind == 'borders':
        for c in mask.columns[:-1]: 
            mask[c] -= mask[c].min()
            mask[c] /= mask[c].max()
        l_ = .5 * (1 - (1-1/step)**(1/d))
        mask.iloc[np.any([mask[c].lt(l_)|\
                          mask[c].gt(1 - l_)\
                          for c in mask.columns[:-1]],
                         axis=0), -1] = 0
    else: 
        raise NotImplementedError("'uniform', 'portion', or 'borders' kind available")
        
    mask.iloc[np.random.choice(mask[mask[val]!=0].index, Spaces.shape[0]//4), -1] = 1
    
    if y is not None and create_dl:
        y = y.reshape(-1, 1)
        DL = []
        for sset in range(3):
            Xs = [torch.tensor(Spaces[mask[val]==sset][:, i].reshape(-1, 1), 
                       requires_grad=True, device=hparams.device
                      ).float() for i in range(Spaces.shape[1])]
            data = (*Xs, torch.tensor(y[mask[val]==sset], requires_grad=True, device=hparams.device))
            # data = TensorDataset(*Xs, torch.tensor(y[mask[val]==sset], device=hparams.device))
            # data = DataLoader(data, batch_size=hparams.batch_size, shuffle=True)
            DL.append(data)
        return DL
    
    return Spaces[mask[val]==0], Spaces[mask[val]==1], Spaces[mask[val]==2]     


def diff(out, inputs):
    inp = inputs[0] if type(inputs) in (list, tuple) else inputs
    return torch.autograd.grad(out, 
                    inputs, 
                    grad_outputs=torch.ones_like(inp),
                    create_graph=True,
                    retain_graph=True,
                   )

##########choose_optimizer

def choose_optimizer(optimizer_name: str, *params):
    if optimizer_name == 'LBFGS':
        return LBFGS(*params)
    elif optimizer_name == 'AdaHessian':
        return AdaHessian(*params)
    elif optimizer_name == 'Adam':
        return Adam(*params)
    elif optimizer_name == 'SGD':
        return SGD(*params)

def LBFGS(model_param,
        lr=1.0,
        max_iter=100000,
        max_eval=None,
        history_size=50,
        tolerance_grad=1e-7,
        tolerance_change=1e-7,
        line_search_fn="strong_wolfe"):

    optimizer = torch.optim.LBFGS(
        model_param,
        lr=lr,
        max_iter=max_iter,
        max_eval=max_eval,
        history_size=history_size,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn=line_search_fn
        )

    return optimizer

def Adam(model_param, lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):

    optimizer = torch.optim.Adam(
                model_param,
                lr=lr,
    )
    return optimizer

def SGD(model_param, lr=1e-4, momentum=0.9, dampening=0, weight_decay=0, nesterov=False):

    optimizer = torch.optim.SGD(
                model_param,
                lr=lr,
                momentum=momentum,
                dampening=dampening,
                weight_decay=weight_decay,
                nesterov=False
    )

    return optimizer

def AdaHessian(model_param, lr=1.0, betas=(0.9, 0.999),
                eps=1e-4, weight_decay=0.0, hessian_power=0.5):
    """
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 0.5)
    """

    optimizer = Adahessian(model_param,
                            lr=lr,
                            betas=betas,
                            eps=eps,
                            weight_decay=weight_decay,
                            hessian_power=hessian_power,
                            single_gpu=False)

    return optimizer


# Get the list of GPUs via nvidia-smi
def free_cuda_id():
    smi_query_result = subprocess.check_output(
        "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
    )
    # Extract the usage information
    gpu_info = smi_query_result.decode("utf-8").split("\n")
    gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
    gpu_info = [int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info]
    return gpu_info.index(min(gpu_info))