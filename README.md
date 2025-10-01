# MoonBoard Grade Prediction Toolkit

## はじめに

このリポジトリは、MoonBoard課題のグレード予測モデルおよび分析ツール一式をまとめたものです。Set Transformer と DeepSet を中心とした順序不変アーキテクチャを PyTorch で実装し、学習・評価・可視化までを Jupyter Notebook ベースで実行できる構成に整理しました。

### フォルダ構成

- `grade_predictor/`  
  グレード分類モデル、データ前処理、推論・可視化ノートブック、出力結果
  - `data/` : 前処理済みMoonBoard課題データとホールド難易度の補助ファイル
  - `analyze/` : 分析用ノートブックと自己評価シート
  - `result/` : 精度ログ、混同行列、外れ値レポートなどの成果物
  - `model.py` / `modules*.py` : Set Transformer・DeepSet 実装と学習ロジック
  - `utils_ordinal.py` : オーディナル回帰用のロス・評価ヘルパー
  - `main.ipynb` : 学習・評価のメインノートブック
- `problem_generator/`  
  今後の課題生成モジュール用ワークスペース（現在はテンプレートのみ）

### 主な特徴

- Set Transformer / DeepSet による順序不変なホールド表現学習
- ホールドタイプ・難易度・XY座標を統合した特徴量設計と加重表現
- グレードの序数構造を活かしたオーディナル回帰ヘッドと専用ロス関数
- 予測誤差が大きい課題の外れ値分析と Excel 形式でのレポート出力
- 混同行列やモデル比較結果を図表・スプレッドシートで可視化

### 必要な環境

- Python 3.9 以上
- PyTorch
- pandas
- scikit-learn
- openpyxl
- matplotlib
- seaborn

必要に応じて `requirements.txt` を整備して `pip install -r requirements.txt` を実行してください。

### 使い方

1. `grade_predictor/main.ipynb` を開き、データ読込・学習・評価セルを順に実行します。
2. 学習後のモデル比較結果や混同行列は `grade_predictor/result/` に自動保存されます。
3. 外れ値調査や追加分析は `grade_predictor/analyze/analyze.ipynb` を利用し、結果を `自己評価.xlsx` に追記してください。
4. 必要に応じて `modules.py` / `modules_modified.py` を編集し、モデルアーキテクチャをカスタマイズできます。

### 出力ファイル

- `grade_predictor/result/accuracy.csv` : モデル別精度サマリー
- `grade_predictor/result/confusion_*.png` : 各モデルの混同行列
- `grade_predictor/result/outlier.xlsx` : 外れ値候補の一覧
- `grade_predictor/result/model_comparison_results.xlsx` : 指標まとめ

`problem_generator/` 以下は今後の開発用スペースです。課題生成ロジックを追加する際は、このディレクトリにノートブックやスクリプトを配置してください。
