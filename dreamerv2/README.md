
# v0.0.1
/logdir/ultimate/2
basic

# v0.0.2
/logdir/ultimate/3
- Add Up Left/Right Special for return stage
- add reward of early killed bonus and combo bonus in env

# v0.0.3 future
- change action to stick and button
    - stick 17 pattern and button abxyzr(xy)(no) 8 pattern
    - all buttons are holded. due to add more frame.
- change frame 6fps in 1.5x, 4fps in 1.0x to 15fps in 1.5x, 10fps in 1.0x

# if change action size and restart variables
tfutils.py line 54 load
```
tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)
to
tf.nest.map_structure(lambda x, y: x.assign(y) if x.shape == y.shape else x, self.variables, values)
```
