# Training Physics-Informed Neural Networks: one learning to rule them all?
Simone Monaco, Daniele Apiletti

Physics-informed neural networks (PINNs) are becoming increasingly popular as powerful instruments to solve nonlinear partial differential equations through deep learning. 
Networks are trained by imposing to respect physical laws incorporated as soft constraints on the loss function. 
Such a basic approach is practical for trivial cases but tends to fail to simulate various classes of more complex dynamical systems. 
Many solutions have been proposed to overcome these limitations. 
However, any interpretation of the problems behind the failures gives rise to different methods for which it is unclear whether they are robust enough to generalise to any system.
Here, we revisit these recent advances, focusing on why the regular training of these methods fails in some benchmark systems. We observed that in some problems, a change in the random initialisation is sufficient to bring the outcome of the network from a complete failure to convergence to the exact solution. On the other hand, some systems instead cannot be solved by the vanilla method at all. Moreover, state-of-the-art improvements that solve some systems' problems fail when applied to others. This sensibly different behaviour reflects how we cannot observe the problems in PINN training from a single perspective.
Therefore, although many novel strategies give promising benefits to PINN results, our experiments prove that there is no one solution to rule them all. This outcome evidences that PINNs are still not well-suited for real-world or industrial applications, but we are not necessarily far from this goal.

## How to run 
An experiment can be launched on an available `SYSTEM` with the desired `TRAINING` by running
```
python pinn_train.py --system <SYSTEM> --training <TRAINING>
```
The default settings are already configured, but some hyperparameters (e.g., epochs, learning rate, etc.) can be customized. More information with
```
python pinn_train.py --help
```

## Citation
The paper has been submitted to "Result in Engineering" journal. It is still under revision.

