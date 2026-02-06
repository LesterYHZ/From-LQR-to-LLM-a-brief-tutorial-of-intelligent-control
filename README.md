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

LQR is a powerful control method for linear systems and thus is quite sufficient to control an inverted pendulum. But to make things more interesting, let's first picture a scenario where we don't have any knowledge about the system model. In this case, all we can do is to try various system inputs and see how the pendulum reacts; the more experience we gain, the better we are able to balance the pendulum. This process is called "*[reinforment learning](https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/)*".  Typicaly we call the input we try on the system "*action*", and the system feedback "*observation*". The action can be discrete, like "*moving the cart to the left or right*", or be continuous, which is quite similar to a traditional controller, computing the control effort based on the system response. We will also define a "*reward*" to evaluate the system performance under the action we applied. The goal of reinforcement learning is to find a "*policy*" that computes the action so that the system results in the highest possible accumulated reward. 

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

However, only applying constant force to move the cart is not efficient to balance the pendulum. Sometimes we would prefer to decide "move the cart slower or faster" to achieve better pendulum balancing performance. Hence, in this case we train the PPO agent on a continuous action space defined by the amplitude of force applied on the cart, just like how LQR controller works. 

First we need to define a reward function that evaluates the system performance. Let's try

$$r=\begin{cases}
    &-10, \quad if \ |\phi|>0.5 \\
    &1-10\phi^2-0.1x^2-0.1\dot{\phi}^2, \quad else
\end{cases}$$

This means that we are training the agent to keep the pendulum angle within $\pm0.5\deg$, and that the smaller the angle, the position, and angular velocity are, the higher reward we get. Using the cart pole equation of motion we made in Chapter 1, we may build a gym-like environment and train the PPO agent in [Cartpole_RL.ipynb](/2.%20Reinforcement%20learning/Cartpole_RL.ipynb). The result is shown below. 

![](/Resources/3.%20Cartpole%20RL.png)



## 3. Imitation learning

Reinforcement learning follows the simple logic of "if you don't succeed, try, try again". But you may have noticed the disadavantages of this method: first, the training take a huge amount of time, and second, the training result highly depends on the design of the reward function. So let's think of an alternative scheme to train an agent: can we directly train the agent to learn from a pro? For example, if we wanna train a learning agent to drive a car, "try, try again" method is almost impossible due to the complexity of the task, but we can let professional drivers to operate the car and record the driver's actions. Then with the data, we may train an agent to directly copy the behavior of the driver, which sounds way more efficient and accessible to train the intelligent model. This process is called "[*imitation learning*](https://smartlabai.medium.com/a-brief-overview-of-imitation-learning-8a8a75c44a9c)". 

### 3.1 Behavioral cloning

[Behavioral cloning](https://www.geeksforgeeks.org/deep-learning/behavioral-cloning/) is the simplest type of imitation learning. Just like how the name sounds, it directly "clones" the behavior of a pro using the collected data. In this section, we use the LQR controller as our pro of balancing the pole. We run LQR multiple times with various initial conditions and record the system state values along with the calculated control efforts to build the dataset. Since this is a quite simple task, let's just build a neural network to train on the dataset. 

The neural network approximate the LQR equation mapping the state variables to control input. As shown blow, the network has nearly 100% the same performance as the LQR controller to keep the pendulum upright. 

![](/Resources/4.%20IL.png)

### 3.2 Residual reinforcement learning

Behaviorial cloning is so much easier compared with reinforcement learning, but a limitation of this method is "*covariate shift*", which is a phenomenon that the distribution of the independent variables in the training and testing data is different. So the performance of a behaviorial cloning agent highly depends on the quality of the dataset and its range of operation. So what if we need the agent to function in wider scenarios? Usually we may apply reinforcement learning to fine-tune the already trained imitation learning agent in order to explore new possible policies and widen its range of operation. But here we want to demonstrate another use of reinforcement learning called [*residual reinforcement learning*](https://ieeexplore.ieee.org/document/8794127), which, instead of learning to directly find the actions, computes the necessary corrections to the actions. A pro's behavior may not be optimal, so we want to apply residual reinforcement learning to "correct" the actions computed by the behavioral cloning agent so that we can maximize the reward we defined. 

[Imitation_learning+Residual_learning.ipynb](/3.%20Imitation%20learning/Imitation_learning+Residual_learning.ipynb) is an example of using residual reinforcement learning to lower the overshoot and undershoot of the behavioral cloning agent imitating the LQR controller. By adding the oscillation and overshoot terms in the reward function, we are able to train a residual reinforcement learning agent that corrects the control effort to receive lower overshoot and eliminate undershoot in the pendulum angle. 

![](/Resources/5.%20IL+RRL.png)

## 4. Language action model

In the recent years, with the rapid development of large model, more and more studies are focusing on integrating vision, language, and actions, which is called a *[VLA](https://learnopencv.com/vision-language-action-models-lerobot-policy/)* model. We don't really need vision to balance the pendulum, so here we demonstrate a language-action model that controls the system via natural languages. Yes, for a simple task like inverted pendulum, this is definitely "killing flies with a cannon", but it's still a fun proof of concept. 


![](/Resources/6.%20Result_dt(0.02)_initial(0.1).png)

![](/Resources/7.%20Result_dt(0.1)_initial(0.2).png)


![](/Resources/8.%20Result_dt(0.02)_initial(-0.3).png)
