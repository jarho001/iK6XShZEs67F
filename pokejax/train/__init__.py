"""Phase 4 training: a recurrent, action-masked PPO on the environment of pokejax.env.

config.py holds the validated configuration, features.py the public damage features,
network.py the policy and value network, ppo.py the jitted update (rollout, GAE, sequence
minibatches), evaluation.py the win rates with Wilson intervals, checkpoint.py the resumable
state and run.py the training loop. Run it with `python -m pokejax.train` (see cli.py); plot.py
draws the win rates of runs (`python -m pokejax.train.plot`).
"""
