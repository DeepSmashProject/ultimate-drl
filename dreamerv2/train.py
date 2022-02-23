import dreamerv2.api as dv2
from env import BaseEnv

config = dv2.defaults.update({
    'logdir': '/workspace/ultimate-drl/dreamerv2/logdir/ultimate/1',
    'log_every': 1e3,
    'train_every': 10,
    'eval_every': 1e3,
    'replay': {'capacity': 2e5, 'ongoing': False, 'minlen': 30, 'maxlen': 50, 'prioritize_ends': True},
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
}).parse_flags()

config2 = dv2.defaults.update({
    'logdir': '/workspace/ultimate-drl/dreamerv2/logdir/ultimate/3',
    'log_every': 1e3,
    'train_every': 10,
    'train_steps': 1,
    'eval_every': 1e3,
    'replay': {'capacity': 2e5, 'ongoing': False, 'minlen': 20, 'maxlen': 50, 'prioritize_ends': True},
    'prefill': 1e4,
}).parse_flags()

env = BaseEnv()
dv2.train(env, config2)

# REQUIRED: apt install ffmpeg
# dreamerv2: line 122,  onstep to onepisode