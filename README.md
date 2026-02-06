# From LQR to LLM: a brief tutorial of intelligent control

![](/Resources/-1.%20Headline.png)

---
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

[LQR_control.ipynb](1.%20LQR%20control/LQR_control.ipynb) showcases the LQR controller balancing the pendulum upright. 

![](Resources/1.%20LQR.png)


## 2. Reinforcement learning

LQR is a powerful control method for linear systems and thus is quite sufficient to control an inverted pendulum. But to make things more interesting, let's first picture a scenario where we don't have any knowledge about the system model. In this case, all we can do is to try various system inputs and see how the pendulum reacts; the more experience we gain, the better we are able to balance the pendulum. This process is called "*[reinforment learning](https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/)*".  Typicaly we call the input we try on the system "*action*", and the system feedback "*observation*". The action can be discrete, like "*moving the cart to the left or right*", or be continuous, which is quite similar to a traditional controller, computing the control effort based on the system response. We will also define a "*reward*" to evaluate the system performance under the action we applied. The goal of reinforcement learning is to find a "*policy*" that computes the action so that the system results in the highest possible reward. 

Here we demonstrate reinforcement learning on both discrete and continuous action space. 

### 2.1 Discrete action

First let's try to balance the pendulum with only two actions: "*push the cart to the left*" and "*push the cart to the right*". [Gymnasium](https://github.com/Farama-Foundation/Gymnasium.git) provides such an [environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/) called "*CartPole-v1*", in which the observation space includes the four state variables we used in LQR chapter -  cart position ($x$), cart velocity ($\dot{x}$), pendulum tilting angle ($\phi$), and pendulum rotating velocity ($\dot{\phi}$). But instead of finding the force applied to the cart, in this case we only consider moving the cart to the left or right. 

In [Gym_cartpole.ipynb](2.%20Reinforcement%20learning/Gym_cartpole.ipynb), we applied [PPO](https://huggingface.co/blog/deep-rl-ppo), or proximal policy optimization, which is probably by far the most powerful reinforcement learning algorithm. You may easily build a PPO agent like playing Legos using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3.git), which contains multiple reliable implementations of differentl reinforcement learning algorithm. It only takes six lines of code to train the agent: 

```Python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("CartPole-v1", n_envs=4)
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
```
Simple, isn't it? And here is how the PPO agent can balance the pendulum after training: 

![Gym cartpole RL](/Resources/2.%20Gym_cartpole.gif)


### 2.2 Continuous action



![](/Resources/3.%20Cartpole%20RL.png)



## 3. Imitation learning

### 3.1 Behavioral cloning

![](/Resources/4.%20IL.png)

### 3.2 Residual reinforcement learning


## 4. Language action model

![](/Resources/6.%20Result_dt(0.02)_initial(0.1).png)

![](/Resources/7.%20Result_dt(0.1)_initial(0.2).png)

![](/Resources/8.%20Result_dt(0.02)_initial(-0.3).png)