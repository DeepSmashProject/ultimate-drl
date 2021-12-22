import dreamerv2.api as dv2
from .env import BaseEnv

config = dv2.defaults.update({
    'logdir': '/workspace/dreamerv2/logdir/rpg/1',
    'log_every': 1e3,
    'train_every': 10,
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
}).parse_flags()
env = BaseEnv()
dv2.train(env, config)