# skylineの使い方

exefsが必要
https://github.com/jugeeya/UltimateTrainingModpack/releases/tag/beta
このbetaを使うのがよさそう

atmosphereの中身をryujinx/mods/contentsの中にいれること。

下記でちぇっくできる
```
cargo skyline set-ip [ubutu, wifiのipアドレス]
cargo skyline listen
```
https://github.com/jugeeya/UltimateTrainingModpack/blob/master/ryujinx_build.sh

### Error
Access Memory region errorが発生する

```
InvalidMemoryRegionException: Attempted to access an invalid memory region.
```
- https://github.com/Ryujinx/Ryujinx-Games-List/issues/2901
  - setting, system tabでignore serviceにチェックをつけるといけるといっているができなかった

よくskylineのログをみるとエラーがあった
```
Error: no offset found for function load_prc_file. refusing to install hook
```

#### UltimateModPackを削除したらエラーが消えた
skylineのログもNo Plungin to loadとなった

- https://github.com/ultimate-research/params-hook-plugin/tags
ここからparams_hookのnroを持ってきてそれを入れたら上記のload_prc_fileエラーがでたものの実行できた。
  - ここは問題ではなさそう
  - [lib_nro_hook](https://github.com/ultimate-research/nro-hook-plugin)
    - nro-hook pluginも問題なく実行できた


## 開発方法
params_hook.nroは必須


### paramsが読み取れない問題
おそらくError: no offset found for function load_prc_fileが原因
prc fileをロードできていないためフックされない
