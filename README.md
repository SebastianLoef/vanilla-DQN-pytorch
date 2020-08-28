# vanilla-DQN-pytorch
My own implementation of the vanilla deep-Q-network using pytorch.
The implementation is designed to be used with OpenAi's atari gym environment,
but any other environments can easily also be used by switching about 
four lines in [train_network](train_network.py) and [test_network](test_network.py).
The training process and hyperparameters are set out to be equivalent 
to DeepMind's: "Human-level control through deep reinforcement learning",
althought i've could missed or missinterpreted something, let me know if 
you find any dissimilarities.

## Requirements
The training process logs runs with tensorboard, this means you 
either need tensorflow for it to work, or comment out the SummaryWriter in 
[train_network](train_network.py) and all it's related code. You can install atari gym
with `pip install 'gym[atari]'`.

