# Deep Reinforscement Learning

For more complex models Reinforcement Learning (RL) can struggle with scale, and Deep Reinforcement Learning (DRL) has been developed to resolve the scalability problem.

## Principles of RL
* Q-Learning
  * [Python example of a simple Q-learning problem][3]
  * The core part is the update of [Q values][2] 
* Temporal Difference (TD) Learning
  * Q-learning is a special case of generalized TD learning. It's one-step TD learning since its learning rate `α=1` 
  * [The core part of updating Q-values in SARSA][4]
  * [Python example using OpenAI gym][5]
* Deep Q-Network
  * Q-table used in TD learning works when in small discrete environment. However when the environment is continuous or has numerious states, it's not practical when it comes to the scalability. Therefore NN is used here to have state as the input and predict the Q-value of each action.
  * The most desirable action has the biggest Q-value
  * [Loss function is MSE][6]
  * The reasons & solutions of unstable Q-network
    * Reason 1 - high correlation between samples due to the sequential nature of sampling experiences
      * Solution - experience replay: the training data is randomly sampled from a buffer of experiences 
        * Prioritized experience replay also suggests that, can't sample uniformly 
    * Reason 2 - non-stationary target is due to the target network Q(s', a') that's modified after every mini btach of training (and therefore there's correlation between adjacent time events)
      * Solution - seperate target Q-network from Q-network under training: copy the params used in Q-network for training under each training steps to target Q-network
   * DQN vs DDQN
     * In DQN, the target Q-network selects and evaluates every action, resulting in an over-estimation of Q-value. Therefore DDQN suggests to use Q-network to select an action and use the target Q-network evaluate that selected action
     * Both DQN and DDQN are able to scale up and solve problems with continuous state space and discrete action space
     * [Python implementation of using DQN, DDQN][7]
* Policy Gradient Methods
  * These methods directly optimize the policy network in RL
  * They are also applicable to environments with discrete and continuous action spaces
  * The goal of the agent is to learn a policy that can maximize the return of all states
  * 4 methods that directly optimize the performance measure of the policy network:
    * REINFORCE, REINFORCE with baseline, Actor-Critic, A2C
      * The author mentioned A3C is the multithreading version of A2C, but A3C is no better when you are using GPUs
    * [Python implementation][8]
      * Gradient ascent if the negative of gradient descent
      * `softplus()` ensures the `std` of the distribution to be positive value
      * `tensorflow_probability` allows you to build the Gaussian distribution and calculate the log probability `log_prob()` of the distribution when given a value
   * Drawbacks
     * Harder to train because of the tendency towards the local optimal instead of the global optimal
     * The policy gradients tend to have high variance
     * Gradient updates are frequently overestimated
     * Time consuming to train


## Reference
* [Advanced Deep Learning with Keras and Tensorflow2][1]


[1]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
[2]:https://play.google.com/books/reader?id=68rTDwAAQBAJ&hl=en_CA&pg=GBS.PA292.w.5.1.17
[3]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/q-learning-9.3.1.py
[4]:https://play.google.com/books/reader?id=68rTDwAAQBAJ&hl=en_CA&pg=GBS.PA306.w.9.0.10.0.2
[5]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/q-frozenlake-9.5.1.py
[6]:https://play.google.com/books/reader?id=68rTDwAAQBAJ&hl=en_CA&pg=GBS.PA313.w.8.0.32.0.1
[7]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/dqn-cartpole-9.6.1.py
[8]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter10-policy/policygradient-car-10.1.1.py
