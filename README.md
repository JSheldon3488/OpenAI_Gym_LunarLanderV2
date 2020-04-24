# OpenAI_Gym_LunarLanderV2
Udacity Deep Reinforcement Learning OpenAI Gym LunarLander-v2 project. Original code and project details can be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn).

## Agents
### Baseline Agent
The baseline agent is a Deep Q-Network with Experience Replay and Fixed Q-Targets. 
More details found in this [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

<p align="center">
    <img src="/images/baseline_Model.png">
</p>

### Baseline + Double DQN
This agent is the same as the baseline agent in every way except Agent.learn now uses the [Double Deep Q-Network](https://arxiv.org/pdf/1509.06461.pdf) algorithm.

<p align="center">
    <img src="/images/Model_Comparison.png">
</p>

### Baseline + Prioritized Experience Replay
This agent is the same as the baseline agent with [prioritized experience replay](https://arxiv.org/pdf/1511.05952.pdf) added.

### Baseline + Dueling DQN

### Baseline + Prioritized Experience Replay and Dueling DQN

## Dependencies
 * [OpenAI Gym](https://gym.openai.com/)
 * [PyTorch](https://pytorch.org/)
 * [Numpy](https://numpy.org/)
 * [box2d](https://box2d.org/)
 * [matplotlib](https://matplotlib.org/)
 #### For rendering on Windows
 * [Xming](https://sourceforge.net/projects/xming/) Full explenation [here](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30).
