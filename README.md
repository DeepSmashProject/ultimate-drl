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