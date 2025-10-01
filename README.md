# MoonBoard Grade Classification & Regression Models

## はじめに

このリポジトリは、MoonBoardの課題グレード分類・回帰モデル（Set Transformer・DeepSetアーキテクチャ）をPyTorchで実装したものです。  
データ前処理、モデル学習、評価、分析スクリプトが含まれています。  
結果や外れ値検出はExcelファイルとして出力され、詳細な確認が可能です。

### フォルダ構成

- `classification_model/`  
  グレード分類モデル、学習スクリプト、分析ツール
- `regression_model/`  
  グレード予測の回帰モデル
- `data/`  
  前処理済みMoonBoard課題データ・ホールド難易度情報
- `result/`  
  出力ファイル（精度ログ、混同行列、外れ値レポート）

### 主な特徴

- Set Transformer・DeepSetによる順序不変学習
- 予測誤差が大きい課題の外れ値検出
- モデル解釈性のためのアテンション可視化
- 結果をExcel形式で出力

### 必要な環境

- Python 3.9以上
- PyTorch
- pandas
- openpyxl
- scikit-learn
- matplotlib
- seaborn

### 使い方

`classification_model/main.ipynb` および `regression_model/main.ipynb` を参照してください。
