# Reinforcement-Learning-Breakout
DRL with DQN PER Dueling for implementation of Atari Game Breakout

This is a Deep Reinforcement Learning using DQN and Q-Learning to approximate the Q-function of Breakout Game via DeepLearning Frame:PyTorch and Reinforcement Learning libs:gym.

Through the feature excration from Convolution Network and Fully Connected Network we can compute the Q-value for each action of the input state.

Using off-policy strategy from Q-learning can we get the model-free intereaction between Agent and Environment.

And using DDQN to fix target network to make the DQN being trained more easily.

We use a ReplayMemory to save the intereaction samples during MDP process.And randomly choice some of them out to train the network.

From current Q-Network and target network can we get TD-Error.

Through loss backward propagation and optimization to the DQN, we can maximize the Agent's performence in Breakout Game.

use python main.py in command line can run the program.You can set running parameters in code yourself.

model.py define the DQN network and its approvement:Dueling DQN

memory.py define the Replay Memory to save the training samples and randomly choice them out.

I add PER imprementation, when the Agent intereact with Environment, it will get the reward, next state, combine with current state, and maximize the
target network, we can get td-error, using td-error to be the priority of each data.After that when we sample the training data out, we using the
priority to normalize to getting the probility of each data.After these data being used to optimize the network, the newly produced td-error of these
data must be updated in Memory Set for them.

Dueling DQN sepreate the Q-function in DQN, it change the fully connected layer to the static value of the current state, in one output dimention and 
action advantage value in current state in num_action dimention. After that we can get the final Q-function is static value add up action advantage value
and minus the mean value of action advantage.

Stable movelize add a shading loss in main function, the 4 frame can produce 4 actions, if two actions nearby is opposite, a shading loss will be output.
