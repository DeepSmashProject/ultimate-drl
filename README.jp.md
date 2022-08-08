# 2022/2/23
## 起動方法
### 0. port 8081, 6006をつけてサーバーに入る
```
ssh -L 6006:xxx.xxx.xxx.xxx:6006 -L 8081:xxx.xxx.xxx.xxx:8081 ruirui@xxx.xxx.xxx.xxx
```

tensorboard
```
cd ~/workspace/DeepSmashProject/ultimate-drl/dreamerv2/logdir
tensorboard --logdir=./ --port 6006 --bind_all
```

### 0. tmux上で作業する
```
tmux new -s ultimate-rl
```

### 1. yuzu_emuをbuild,起動する
```
cd workspace
git clone https://github.com/DeepSmashProject/yuzulib.git
cd yuzulib/docker
bash build.sh
bash run.sh
```
localhost:8081からnovncを確認できる。passwordは「pass」

### 2. ultimate-drl, ultimate-gymをinstallする
// in yuzu_emu container
```
cd /workspace
git clone https://github.com/DeepSmashProject/ultimate-gym.git
git clone https://github.com/DeepSmashProject/ultimate-drl.git
pip install -e yuzulib -e ultimate-gym
```

### 3. スマブラを起動する
スマブラのホーム, トレーニングまでの起動
```
cd yuzulib/scripts
bash run_ssbu.sh
bash move_to_home.sh
bash move_to_training.sh
```
envテスト
```
cd ultimate-gym/scripts
bash env_test.sh
```
screen: 214, 141, 853, 441となっているか確認

### 4. dreamerv2をinstallしてコードを書き換える

```
cd /workspace
git clone https://github.com/danijar/dreamerv2.git
vim dreamerv2/dreamerv2/api.py
```
111行目
```
def train_step(train, worker): -> def train_step(ep):
```
122行目
```
driver.on_step(train_step) -> driver.on_episode(train_step)
```
onstepだとstepごとにtrainが入り、スマブラの操作ができなくなるため。

```
cd /workspace
pip install -e dreamerv2
```

### 5. ffmpegのinstall
```
apt update
apt install ffmpeg
```

### 6. dreamerv2の設定、実行
```
cd ultimate-drl/dreamerv2
vim train.py
```

configの変更
```
config = dv2.defaults.update({
    'logdir': '/workspace/ultimate-drl/dreamerv2/logdir/ultimate/2022-02-23',  <-日付に変更
    'log_every': 1e3,
    'train_every': 10,
    'train_steps': 1,
    'eval_every': 1e3,
    'replay': {'capacity': 2e5, 'ongoing': False, 'minlen': 20, 'maxlen': 50, 'prioritize_e
nds': True},
    'prefill': 1e4,
}).parse_flags()
```

実行
```
python3 train.py
```

# Error

### Tensorflow error

```
    _descriptor.FieldDescriptor(
  File "/usr/local/lib/python3.8/dist-packages/google/protobuf/descriptor.py", line 560, in __new__
    _message.Message._CheckCalledFromGeneratedFile()TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.If you cannot immediately regenerate your protos, some other possible workarounds are: 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
```

```
ImportError: cannot import name 'experimental' from 'tensorflow.keras.mixed_precision'
```

### 解決法
```
https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/nets.py
from tensorflow.keras.mixed_precision import experimental as prec
to
import tensorflow.keras.mixed_precision as prec

```

# Memo

### /ultimate/3
- UP_RIGHT_SPECIAL, UP_LEFT_SPECIALを追加した

### /ultimate/4
- 2022/02/23~再び学習開始
- rewardにコンボとearlykillボーナスを追加
- 他は3と同じで3の続きから学習している
- serverはosiris

### /ultimate/20220228_clipping
- 報酬を0,1にクリッピングしてためす。
- serverはhorus

### /ultimate/20220301_clipping_init_actor_critic
- 報酬をクリッピングして、かつactor_criticの重みをはじめから学習した
- 以前のに引っ張られて同じ動きしかしなかったため。




```
['Variable:0', 'Variable:0', 'dense_16/kernel:0', 'dense_16/bias:0', 'dense_17/kernel:0', 'dense_17/bias:0', 'dense_18/kernel:0', 'dense_18/bias:0', 'dense_19/kernel:0', 'dense_19/bias:0', 'dense_20/kernel:0', 'dense_20/bias:0', 'dense_11/kernel:0', 'dense_11/bias:0', 'dense_12/kernel:0', 'dense_12/bias:0', 'dense_13/kernel:0', 'dense_13/bias:0', 'dense_14/kernel:0', 'dense_14/bias:0', 'dense_15/kernel:0', 'dense_15/bias:0', 'dense_21/kernel:0', 'dense_21/bias:0', 'dense_22/kernel:0', 'dense_22/bias:0', 'dense_23/kernel:0', 'dense_23/bias:0', 'dense_24/kernel:0', 'dense_24/bias:0', 'dense_25/kernel:0', 'dense_25/bias:0', 'Variable:0', 'conv2d/kernel:0', 'conv2d/bias:0', 'conv2d_1/kernel:0', 'conv2d_1/bias:0', 'conv2d_2/kernel:0', 'conv2d_2/bias:0', 'conv2d_3/kernel:0', 'conv2d_3/bias:0', 'conv2d_transpose/kernel:0', 'conv2d_transpose/bias:0', 'conv2d_transpose_1/kernel:0', 'conv2d_transpose_1/bias:0', 'conv2d_transpose_2/kernel:0', 'conv2d_transpose_2/bias:0', 'conv2d_transpose_3/kernel:0', 'conv2d_transpose_3/bias:0', 'dense/kernel:0', 'dense/bias:0', 'dense_6/kernel:0', 'dense_6/bias:0', 'dense_7/kernel:0', 'dense_7/bias:0', 'dense_8/kernel:0', 'dense_8/bias:0', 'dense_9/kernel:0', 'dense_9/bias:0', 'dense_10/kernel:0', 'dense_10/bias:0', 'dense_1/kernel:0', 'dense_1/bias:0', 'dense_2/kernel:0', 'dense_2/bias:0', 'dense_3/kernel:0', 'dense_3/bias:0', 'dense_4/kernel:0', 'dense_4/bias:0', 'dense_5/kernel:0', 'dense_5/bias:0', 'dense/kernel:0', 'dense/bias:0', 'layer_normalization/gamma:0', 'layer_normalization/beta:0', 'dense_2/kernel:0', 'dense_2/bias:0', 'dense/kernel:0', 'dense/bias:0', 'dense_1/kernel:0', 'dense_1/bias:0', 'dense_1/kernel:0', 'dense_1/bias:0', 'dense/kernel:0', 'dense/bias:0']
Skipping short episode of length 15.
Episode has 15 steps and return -1.0.
```
