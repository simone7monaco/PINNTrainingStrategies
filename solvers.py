import numpy as np
from utils import function
from scipy.integrate import odeint


def burg_system(u, t, k, nu):
    #Spatial derivative in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_x = 1j*k*u_hat
    u_hat_xx = -k**2*u_hat
    
    #Switching in the spatial domain
    u_x = np.fft.ifft(u_hat_x)
    u_xx = np.fft.ifft(u_hat_xx)
    
    #ODE resolution
    u_t = -u*u_x + nu*u_xx
    return u_t.real


def rd_system(u, t, k, nu, rho):
    #Spatial derivative in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_xx = -k**2*u_hat
    
    #Switching in the spatial domain
    u_xx = np.fft.ifft(u_hat_xx)
    
    #ODE resolution
    u_t = nu*u_xx + rho * u * (1 - u)
    return u_t.real


def ac_system(u, t, k, nu, rho):
    #Spatial derivative in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_xx = -k**2*u_hat
    
    #Switching in the spatial domain
    u_xx = np.fft.ifft(u_hat_xx)
    
    #ODE resolution
    u_t = -rho * (u**3 - u) + nu*u_xx
    return u_t.real


def fft2_discrete_sol(x, t, u0 : str, system, **sys_args):
    #Wave number discretization
    N_x = len(x)
    dx = (x[-1] - x[0])/N_x

    k = 2*np.pi*np.fft.fftfreq(N_x, d=dx)

    #Def of the initial condition
    u0 = function(u0)(x)
    
    if system == 'burger':
        Ft = burg_system
    elif system == 'allencahn':
        Ft = ac_system
    elif system == 'reaction-diffusion':
        Ft = rd_system 
    else:
        raise NotImplementedError("System not valid.")
    return odeint(Ft, u0, t, args=(k, *sys_args.values()),
                  mxstep=5000)  


def convection_diffusion(x, t, u0 : str, nu, beta, rho, source=0, nx=256, nt=100):
    """Calculate the u solution for convection/diffusion, assuming PBCs.
    Args:
        u0: Initial condition
        nu: viscosity coefficient
        beta: wavespeed coefficient
        source: q (forcing term), option to have this be a constant
        xgrid: size of the x grid
    Returns:
        u_vals: solution
    """

    N = nx
    X, T = np.meshgrid(x, t)

    # call u0 this way so array is (n, ), so each row of u should also be (n, )
    u0 = function(u0)(x)

    G = np.zeros_like(u0) + source # G is the same size as u0

    IKX_pos =1j * np.arange(0, N/2+1, 1)
    IKX_neg = 1j * np.arange(-N/2+1, 0, 1)
    IKX = np.concatenate((IKX_pos, IKX_neg))
    IKX2 = IKX * IKX

    uhat0 = np.fft.fft(u0)
    nu_factor = np.exp(nu * IKX2 * T - beta * IKX * T)
    A = uhat0 - np.fft.fft(G)*0 # at t=0, second term goes away
    uhat = A*nu_factor + np.fft.fft(G)*T # for constant, fft(p) dt = fft(p)*T
    u = np.real(np.fft.ifft(uhat))

    u_vals = u.flatten()
    return u_vals