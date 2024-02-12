
# 7layer-U-Net++
<img src="https://github.com/RyoTakeshita0910/IVUS-2024-DL/assets/104045526/c57229c8-0b92-4ee3-931b-5fb376514643.png" width="500">

#### 主に使用するファイル
・arch.py：深層学習モデルのネットワーク構造<br>
・train.py：訓練の実行プログラム<br>
・val.py：テストの実行プログラム<br>

#### 基本置いておくだけのファイル
・losses.py：損失関数(今後、変更をする場合はあるかも？)<br>
・dataset.py：データの読込方法<br>
・metrics.py：accuracy(Dice係数)の計算式<br>
・utils.py：関数<br>
・preprocess_dsb2018.py：デモ用プログラム？(マジで使わん)<br>


## 環境設定
・anacondaの仮想環境の作成<br>
・CUDA：ver10.0
```bash
conda env create --file requirements.yaml
```

## train
### trainのデータセット
・フォルダ名やフォルダの場所は**必ず**下の例と同一にする<br>
・入力画像(images)とラベル画像(masks)の画像ファイルの名前は、**必ず**一致させる<br>
・masksの画像フォルダの数はクラス数と一致させる(ex：class_num=1の場合、フォルダは「0」のみ)
```bash
inputs
└── <dataset name>
    ├── images
    │   ├── xxx_00000x.png
    │   ├── yyy_00000y.png
    │   ├── ...
    │
    └── masks
        ├── 0 
        │   ├── xxx_00000x.png
        │   ├── yyy_00000y.png
        │   ├── ...
        │   
        ├── 1 
        │   ├── xxx_00000x.png
        │   ├── yyy_00000y.png
        │   ├── ...

```

### trainの実行方法
#### 必須のコマンドライン引数
・dataset：inputs内のデータセット名を指定<br>
・arch：学習に用いるネットワークの名前を以下から指定<br>
　UNet, NestedUNet, NestedUNet7, DPUNet, NestedDPUNet<br><br>
・batch size：NestedUNet7を用いる際は**2**にする(変えたときにプラーク分類の精度が落ちたため)<br>
・num_classes：Lumen,Media,wireの学習では**1**, プラークの分類では**6**にする<br>

#### 任意のコマンドライン引数
・deep_supervision：**Deep supervision**という機能を搭載するかどうか<br>
　引数を"True"にするとDeep supervisonを搭載<br>
・epochs：エポック数(default:100)<br>
・optimizer：最適化アルゴリズム(default：SGD)．必要であれば，引数を"Adam"にするとAdamに変更可能．<br>

#### データごとの学習設定
・Lumen：deep supervision あり<br>
・Media,Wire,プラークの分類：deep supervisionなし

```bash
python train.py --dataset inputs内の(dataset name) --arch Network Name(default:NestedUNet7) -b batch size(default:2) --num_classes クラス数(default:1)
```

### train後の処理
・modelsに学習結果が保存される．<br>
保存時のフォルダ名は，以下のようになる<br>
・Deep supervisionを使用しない：(dataset_name)_(network_name)_woDS<br>
・Deep supervisionを使用する：(dataset_name)_(network_name)_wDS<br>

## test
### testのデータセット
・trainと同様のデータ配置にする<br>
・masksにはテスト画像の正解マスクを保存．マスク画像は，正解と予測結果のDice係数を計算するために使用．
```bash
test_inputs
└── <dataset name>
    ├── images
    │   ├── xxx_00000x.png
    │   ├── yyy_00000y.png
    │   ├── ...
    │
    └── masks
        ├── 0 
        │   ├── xxx_00000x.png
        │   ├── yyy_00000y.png
        │   ├── ...
        │   
        ├── 1 
        │   ├── xxx_00000x.png
        │   ├── yyy_00000y.png
        │   ├── ...

```

### testの実行方法
・name：modelsにある学習結果のフォルダの名前<br>
・input：test_inputsの中にあるテスト用データセットのパスを指定
```bash
python val.py --name 学習済み重みの指定 --input ~/test/dataset/path
```

### test後の処理
・outputsに学習結果のフォルダ名で推論結果を保存

## シェルファイル(.sh)による実行
シェルファイルによる実行の自動化

・シェルファイルの中身
```bash
#!usr/bin/bash

python train.py ~
python train.py ~
...

pause
```

・シェルファイルの実行コマンド
```bash
sh (ファイル名).sh
```
