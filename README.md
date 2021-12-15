# ultimate-drl

Version 1 (Odin):

# Setup
pip install -r requirements.txt
cd yuzulib && pip install -r requirements.txt
cd libultimate && pip install -r requirements.txt
cd ultimate-gym && pip install -r requirements.txt
cd ultimate-drl && pip install -r requirements.txt
cd utils/yolov5 && pip install -r requirements.txt

# Run
python3 train.py -f v0_develop/v001_damage_rainbow.yaml

# Restore
python3 train.py -f v0_develop/v001_damage_rainbow.yaml -r /workspace/ultimate-drl/results/v0.0.0_base_damage_rainbow/DQN_BaseEnv_6b62d_00000_0_2021-11-28_03-58-17/checkpoint_000290/checkpoint-290

# Attention
screen size is 213 141 853 487 or 214 141 853 487 only 

# tensorboard
docker exec -it yuzu_emu bash
cd ultimate-drl/results/v1.0.0_base_4ch_rainbow/
tensorboard --logdir . --port 6007 --bind_all


# User Warning
 /usr/local/lib/python3.8/dist-packages/ray/rllib/agents/dqn/dqn_torch_model.py:151: UserWarning: torch.range is deprecated and will be removed in a future release because its behavior is inconsistent with Python's range builtin. Instead, use torch.arange, which produces values in [start, end).

/usr/local/lib/python3.8/dist-packages/ray/rllib/agents/dqn/dqn_torch_model.py
```
z = torch.range(
                0.0, self.num_atoms-1,
                dtype=torch.float32).to(action_scores.device)
```
to
```
z = torch.arange(
    0.0, self.num_atoms,
    dtype=torch.float32).to(action_scores.device)

```

/usr/local/lib/python3.8/dist-packages/ray/rllib/agents/dqn/dqn_torch_policy.py:50: UserWarning: torch.range is deprecated and will be removed in a future release because its behavior is inconsistent with Python's range builtin. Instead, use torch.arange, which produces values in [start, end).

```
 z = torch.range(
                0.0, num_atoms-1, dtype=torch.float32).to(rewards.device)

 z = torch.arange(
                0.0, num_atoms, dtype=torch.float32).to(rewards.device)
```