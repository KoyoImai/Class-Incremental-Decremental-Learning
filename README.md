# Class-Incremental-Decremental-Learning (CIDL)
クラス増加かつクラス減少学習のためのプログラムです．

# プログラムの全体像
学習・評価に使用するプログラムの全体像は以下の通りです．
```
SSOCL/
├── augmentations     : データ拡張関連を実装したモジュール群
├── configs           : 学習・評価の設定を記述する.yamlファイルの格納場所．
├── dataloaders       : DataLoader関連を実装したモジュール群．
├── losses            : 損失関数関連を実装したモジュール群．
├── models            : model関連を実装したモジュール群．
├── optimizers        : Optimizer関連を実装したモジュール群．
├── train             : 訓練・評価を実際に行うモジュール群．
├── main.py           : データストリームによる事前学習を実行するmainファイル．
├── 
├── 
└── utils.py          : その他のモジュールを実装するutilsファイル．
```


# 実行方法
学習・評価の実行方法は以下の通りです．
- Finetune: \
    忘却への対処やUnlearning的なアプローチを行わない通常の学習です．
    ```
    python main.py --config-path ./configs/default/ --config-name finetune
    ```

- LSF ([paper](https://www.ijcai.org/proceedings/2021/0137.pdf)):\
    従来手法LSFによる学習です．
    ```
    python main.py --config-path ./configs/default/ --config-name lsf
    ```