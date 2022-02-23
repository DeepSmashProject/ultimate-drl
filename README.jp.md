# 2022/2/23
## 起動方法
### 0. port 8081, 6006をつけて研究室サーバーに入る
https://ruihirano.atlassian.net/wiki/spaces/ZENTAI/pages/384696321
```
ssh -L 6006:192.168.207.237:6006 6007:192.168.207.237:6007 ruirui@133.6.254.20
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


# Memo

### /ultimate/3
- UP_RIGHT_SPECIAL, UP_LEFT_SPECIALを追加した
