# PAC-Bayesian Generalization Bounds for Adversarial Generative Models 再現コード

このリポジトリは，大学院講義の期末課題にて，論文「PAC-Bayesian Generalization Bounds for Adversarial Generative Models」の再現実験を試みたコードです．

## 論文

- [PAC-Bayesian Generalization Bounds for Adversarial Generative Models (arXiv:2302.08942)](https://arxiv.org/abs/2302.08942)

## 動作環境

- [uv](https://github.com/astral-sh/uv) パッケージマネージャを使用
- CUDA 12.1のGPU環境で動作させた

## セットアップと実行方法

### 実行手順

1.  `uv sync` を実行し環境をつくります
2.  以下のコマンドで実行します

```powershell
uv run python -m src.train_synthetic --dataset Ring --epochs 30 --pretrain-epochs 2 --mc-samples-train 10 --mc-samples-eval 100 --outdir out
uv run python -m src.train_synthetic --dataset Grid --epochs 30 --pretrain-epochs 2 --mc-samples-train 10 --mc-samples-eval 100 --outdir out
```