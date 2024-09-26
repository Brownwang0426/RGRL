# RGRL
## Introduction
We introduce **R**eversal **G**enerative **R**einforcement **L**earning (**RGRL**), a **model-free** and **value-function-free** reinforcement learning method, where **only** neural networks are involved.

**RGRL** treats an agent as ***a neural net ensemble*** where ***state*** and ***action*** are ***input*** datum, and ***reward*** is ***output*** data. First, when the neural net ensemble (the agent) observes certain state, it initializes and updates its actions through error backpropagation, with given desired reward as target. Second, the agent executes the updated actions and learns the ensuing consequence (state, executed actions and actual reward). Third, based on the learned neural net ensemble, the agent starts observing new state and initializing or updating its actions again. Under this iterative updating-then-learning process, the agent gradually forms a belief approximating the real environement, allowing the agent to find the optimal solution to achieve the maximum reward in the environment. This approach is very similar to reinforcement learning in our intuitive finding. Since action is gradually generated from input layer rather than output layer, we refer to this method as **R**eversal **G**enerative **R**einforcement **L**earning (**RGRL**).

In **RGRL**, we use deep neural nets to substitute Bellman function to provide more flexibility. That is to say, we use deep neural nets to map future reward to present state and actions or to predict future reward using present state and actions.

Our previous research can be seen in this [paper](https://ala2022.github.io/papers/ALA2022_paper_4.pdf), where we simply used supervised learning method rather than reinforcement learning method. However, in **RGRL**, we have inherited the spirit of the previous research while incorporating the concept of reinforcement learning, allowing the agent to learn from the ensuing consequence of it deduced or updated actions. In practice, this enables the agent to find the optimal solution to achieve the maximum reward in the environment more quickly.

## Features
- **Neural Nets are all you need**: Seriously, really.
- **Highly custimizable**: All you need to do is to customize state, action and reward-shaping or vectorizing. You can also make state as raw input if you prefer :-) Why not? It is deep neural network :-) 

## Future Works
- **Online learning**: For the present time, we present offline learning version of RGRL.

## Getting Started
To get started with RGRL, follow these steps:

1. **Open the .ipynb in colab and select T4 GPU or above**
2. **Ctrl + F10, restart and Ctrl + F10**
3. **Take a rest and wait for the result**

## Experimental Results
We use traditional **Cartpole** as an example and show that how the size of an ensemble of neural networks affect the overall performace of the agent.

## Why an ensemble of neural networks rather than a single neural network?

Building on our previous research in this [paper](https://ala2022.github.io/papers/ALA2022_paper_4.pdf), we observed that when using error backpropagation to update input data for model inversion in a trained deep neural network—similar to techniques in DeepDream—the updated input data often becomes unstable and gets stuck. After multiple trials, we identified that this instability occurs because the input is essentially performing gradient descent on a single error surface generated by one deep neural network, leading to numerous local minima, regardless of the network’s level of training.

To mitigate this instability, we borrow the concept of traditional stochastic gradient descent method, where input data and labels are shuffled to train a neural network. However, the difference now is that we shuffled an ensemble of trained deep neural networks during the input data update, preventing the input from getting trapped in local minima.

In our earlier work, we demonstrated the effectiveness of this method, achieving a 97.3% success rate in solving blank Sudoku puzzles using the ensemble of deep neural networks.

## Project Structure

We try to keep the structure as clean and easy as possible. The whole structure of RGRL is as follows:

```bash
RGRL/
│
├── models/                  # Model-related files or classes
│   ├── __init__.py
│   ├── model.py             # Model implementation
│   └── model_utils.py       # Helper functions for the model
│
├── utils/                   # General utility functions
│   ├── logging.py           # Logging system
│   ├── metrics.py           # Metrics calculation
│   └── helpers.py           # Common helper functions
│
├── in_experiment/           # Models in exepriment. Not ready, but you may try it out!
│
├── train.py                 # Main script for training the model
│
└── README.md                # Project documentation
```

## Status
The project is currently in active development. We are continually adding new features and improving the performance.

## License
RGRL is released under the [MIT](https://github.com/Brownwang0426/RGRL/blob/main/LICENSE) license.

## Related Works
- 
- 





