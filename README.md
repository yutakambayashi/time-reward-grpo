# 時間依存報酬 GRPO による VLA の作業サイクルタイム短縮

模倣学習済みの VLA モデル SmolVLA を、速く成功するほど高い報酬を与える GRPO で追加学習し、LIBERO-Object のタスクをより短い時間でこなせるようにする実験のコードと評価結果です。Physical AI 応用編1 講座の最終課題として作成しました。

## 結果

評価には学習に使っていない初期状態（10タスク × 10、計100エピソード）を使いました。速さは、比べる2つのモデルが両方とも成功した初期状態どうしで、成功までのステップ数 $T$ を比べています（両側の符号検定）。

| モデル | 成功数（/100） | 所要ステップ数 |
|---|---|---|
| SFT（追加学習なし） | 79 | 基準 |
| 成功のみの報酬で40バッチ | 85 | SFT より遅い（70件中59件で遅い、差の中央値 +9.5） |
| 時間依存報酬で40バッチ | 81 | SFT より速い（68件中56件で速い、差の中央値 −12、p = 5.6×10⁻¹⁰） |

時間依存報酬は成功のみの報酬と比べても、77件中71件で速くなりました（差の中央値 −22、p = 6.8×10⁻¹⁷）。40バッチ目の3つのモデルの成功数には、どの組み合わせでも有意な差はありません。数値はすべて `results/final_stats.txt` にあります。

## 手法

報酬は次のとおりです。$T$ は成功までのステップ数、$T_{\max} = 280$ はエピソードの上限で、$\lambda = 1$ としました。

$$R = \mathbb{1}[\text{成功}]\left(1 + \lambda\left(1 - \frac{T}{T_{\max}}\right)\right)$$

成功の報酬は 1 以上 2 未満、失敗は 0 なので、どんなに遅い成功も失敗より報酬が高くなります。成功のみの報酬では、全試行が成功したグループは報酬に差がつかず学習から外れる一方、時間依存報酬ではこうしたグループでも所要時間の差が学習の信号になります。

学習には [ART-Embodied](https://github.com/wandb/ART-Embodied) の Flow-SDE GRPO を使いました。1バッチは1つの初期状態から8回試行した組（グループ）を4つ集めた32エピソードで、1バッチごとに方策を1回更新します。

## リポジトリの構成

| パス | 内容 |
|---|---|
| `patches/patch_time_reward.py` | ART-Embodied に時間依存報酬、所要ステップ数の記録、グループの絞り込みを追加するパッチ |
| `scripts/make_cfg_train.py` | 学習と評価の設定ファイルを書き出す |
| `scripts/collect_rewards.py` | 学習中の各試行の報酬と成否を CSV にまとめる |
| `scripts/final_stats.py` | 報告書の数値（成功数、対応のある比較、タスク別の結果、学習に使ったグループ）を計算する |
| `scripts/select_scenarios.py`, `scripts/build_compare.py` | 比較動画の初期状態を選び、3つのモデルの映像を横に並べる |
| `scripts/summarize.py` | 評価結果1つをタスク別に要約する |
| `notebooks/01_train_grpo.ipynb` | 追加学習（報酬の条件を切り替えて2回実行する） |
| `notebooks/02_eval_sft_and_videos.ipynb` | SFT の評価と比較動画の作成 |
| `results/eval/` | 各モデルの評価結果（`episode_outcomes.json`、各100エピソード） |
| `results/reward_evidence.csv` | 学習中の全試行の報酬と成否（各条件40バッチ × 32試行） |
| `results/final_stats.txt` | `final_stats.py` の出力 |

`results/eval/` のファイルと、報告書で使った版の対応は次のとおりです。

| ファイル | 版 |
|---|---|
| `sft.json` | SFT（追加学習なし） |
| `success_b10.json`, `success_b20.json`, `success_b40.json` | 成功のみの報酬の10・20・40バッチ目 |
| `time_b20.json`, `time_b40.json` | 時間依存報酬の20・40バッチ目 |

## 再現の手順

### 報告書の数値を計算し直す（GPU 不要）

```bash
python3 scripts/final_stats.py > results/final_stats.txt
```

Python 3.9 以降の標準ライブラリだけで動きます。

### 学習と評価をやり直す（Google Colab の L4 GPU）

1. `notebooks/01_train_grpo.ipynb` を Colab で開く（先頭のセルの `REPO_URL` は、フォークした場合はその URL に変える）
2. `CONDITION = "time"` で全セルを実行する。学習は裏で動くので、状態確認のセルで進み具合を見る（約6時間）
3. `CONDITION = "success"` にして同じように実行する
4. `notebooks/02_eval_sft_and_videos.ipynb` で SFT を評価し、比較動画を作る（約1時間10分）
5. 学習中の報酬を集計するには、Colab で `scripts/collect_rewards.py` を実行する

   ```bash
   python3 scripts/collect_rewards.py results/reward_evidence.csv S=$OUT_ROOT/grpo-success T=$OUT_ROOT/grpo-time
   ```

評価は乱数を使わないので、同じチェックポイントからは同じ結果が出ます。学習には乱数が入るため、学習をやり直したときの結果は本リポジトリの値と一致するとは限りません。

## 動作環境

- ART-Embodied commit `a37617ebfd9ced9455b6d4527a800f632fd0e37b`（パッチはこの commit に合わせて書いています）
- モデル: `HuggingFaceVLA/smolvla_libero`。行動エキスパートに LoRA（rank 64）を付けて学習
- 行動生成: 生成の段数 4、Flow-SDE のノイズ 0.5。50ステップ分を予測し、先頭10ステップを実行
- 計算資源: Google Colab の NVIDIA L4 1枚（ハイメモリ）。1バッチの学習に約8〜10分、100エピソードの評価に約30分

## 注意

- 実験は整理する前のノートブックとパッチで実行しました。整理では使わなかった機能（物体の持ち上げを検知して加点する報酬）を削除し、設定を書き出すスクリプトを短くしました。整理後のパッチは指定の commit に当たること、`results/eval/` に記録された報酬をすべて同じ値に再現することを確かめています。整理後の設定ファイルは、整理前のものと使わなかった報酬の項目以外は一致します。整理後のノートブックを通して実行したことはありません。
- `results/eval/` の各ファイルには、整理前のパッチが書き出した `lift_step`（物体を持ち上げたステップ）と `regrasps` の列が残っています。どちらも報酬には使っていません（持ち上げの加点の重みは 0）。

## ライセンス

本リポジトリのコードは MIT License（`LICENSE`）です。ただし、`patches/patch_time_reward.py` に含まれる [ART-Embodied](https://github.com/wandb/ART-Embodied) のソースの断片（置き換え元の文字列）は ART-Embodied のもので、[Apache License 2.0](https://github.com/wandb/ART-Embodied/blob/main/LICENSE) に従います。ART-Embodied 本体は同梱しておらず、実行時に公式リポジトリから取得します。
