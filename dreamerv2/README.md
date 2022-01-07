
# v0.0.1
/logdir/ultimate/2
basic

# v0.0.2
/logdir/ultimate/3
- Add Up Left/Right Special for return stage

# if change action size and restart variables
tfutils.py line 54 load
```
tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)
to
tf.nest.map_structure(lambda x, y: x.assign(y) if x.shape == y.shape else x, self.variables, values)
```
