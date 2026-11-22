"""The trained policy played on Showdown (Phase 5).

reference/showdown-player.cjs plays many battles at once in the pinned simulator. At each learner
decision it sends the observation of reference/player-view.cjs to this package, which runs the
network on the batch of waiting battles and returns one action per battle. bridge.py is the line
protocol, player.py the batched policy and the run loop, __main__.py the driver and compare.py the
comparison with the engine's evaluation.
"""
