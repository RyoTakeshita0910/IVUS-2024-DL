
# 7layer-U-Net++

### 主に使用するファイル
・arch.py：深層学習モデルのネットワーク構造<br>
・tarin.py：訓練の実行プログラム<br>
・val.py：テストの実行プログラム<br>

### 基本置いておくだけのファイル
・losses.py：損失関数(今後、変更をする場合はあるかも？)<br>
・dataset.py：データの読込方法<br>
・metrics.py：accuracy(Dice係数)の計算式<br>
・utils.py：関数<br>
・preprocess_dsb2018.py：デモ用プログラム？(マジで使わん)<br>


## 環境設定
・anacondaの仮想環境の作成<br>
・CUDA：ver10.0
```bash
conda env export > requirements.yaml
```

## trainのデータセット
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

## trainの実行方法
・dataset：inputs内のデータセット名を指定<br>
・arch：学習に用いるネットワークの名前を以下から指定<br>
　**UNet, NestedUNet, NestedUNet7, DPUNet, NestedDPUNet<br>
・batch size：NestedUNet7を用いる際は**2**にする(変えたときにプラーク分類の精度が落ちたため)<br>
・num_classes：Lumen,Media,wireの学習では**1**, プラークの分類では**6**にする<br>
```bash
python train.py --dataset inputs内の(dataset name) --arch Network Name(default:NestedUNet7) -b batch size(default:2) --num_classes クラス数(default:1)
```

