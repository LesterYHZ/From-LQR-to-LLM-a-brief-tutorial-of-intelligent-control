# From LQR to LLM: a brief tutorial of intelligent control

Robots are becoming more and more intelligent in these days and ages, thanks to the rapid development of computing units and learning algorithms. This tutorial serves as an amature demonstration of intelligent control evolving from simple optimal control to reinforcemnt learning and eventually to LLM-driven, using the most classic model in control engineering - inverted pendulum. 



## 1. LQR control
This is probably the first multi-variable control project that every control engineer worked on in college. The cart pole system is shown below. The system consists of an inverted pendulum mounted to a motorized cart. The pendulum will simply fall over if the cart isn't moved to balance it. Additionally, the dynamics of the system are nonlinear. The objective of the control system is to balance the inverted pendulum by applying a force to the cart that the pendulum is attached to. 

![A cart-pole inverted pendulum.](Resources/0.%20cart-pole.png) 

Applying Lagrange's equation can we write the dynamic model of the system: 

$$(M+m)\ddot{x}+b\dot{x}+ml\ddot{\theta}\cos\theta-ml\dot{\theta}^2\sin\theta=F$$

$$(I+ml^2)\ddot{\theta}+mgl\sin\theta=-ml\ddot{x}\cos\theta$$

This model can be linearized to bulid a state space model involving the cart position ($x$), cart velocity ($\dot{x}$), pendulum tilting angle ($\phi$), and pendulum rotating velocity ($\dot{\phi}$). 

$$\frac{d}{dt}\begin{pmatrix}
    {x} \\\ \dot{x} \\\ \{\phi} \\\ \dot{\phi}
\end{pmatrix} = \begin{pmatrix}
    0 & 1 & 0 & 0 \\\
    0 & \frac{-(I+ml^2)b}{I(M+m)+Mml^2} & 
    \frac{m^2gl^2}{I(M+m)+Mml^2} & 0 \\\
    0 & 0 & 0 & 1 \\\
    0 & \frac{-mlb}{I(M+m)+Mml^2} & \frac{mgl(M+m)}{I(M+m)+Mml^2} & 0 
\end{pmatrix}\begin{pmatrix}
    x \\\ \dot{x} \\\ \phi \\\ \dot{\phi}
\end{pmatrix} + \begin{pmatrix}
    0 \\\ 
    \frac{I+ml^2}{I(M+m)+Mml^2} \\\ 
    0 \\\
    \frac{ml}{I(M+m)+Mml^2}
\end{pmatrix}F$$

And for simplicity, from now on, we will write the equation of motion as

$$\dot{X}=AX+BU$$

Since we are doing optimal control, let's first define a cost function

$$J = \int_0^\infty(X^TQX+U^TRU)\;dt$$

where $Q$ and $R$ are the state and control cost matrices. 

LQR, or linear–quadratic regulator, finds the control input by solving the algebraic Riccati equation for $P$: 

$$A^TP+PA-PBR^{-1}B^TP+Q=0$$

and then the control input can be calculated by

$$U = -R^{-1}B^TPX$$

You may also use the [Python Control Systems Library
](https://github.com/python-control/python-control.git) to implement the control algorithm. 

[LQR_control.ipynb](1.%20LQR%20control/LQR_control.ipynb) in [Chapter 1](1.%20LQR%20control/)'s folder showcases the LQR controller balancing the pendulum upright. 

![](Resources/1.%20LQR.png)



## 2. Reinforcement learning

### 2.1 Discrete action space

![Gym cartpole RL](/Resources/2.%20Gym_cartpole.gif)


### 2.2 Continuous action

![](/Resources/3.%20Cartpole%20RL.png)
