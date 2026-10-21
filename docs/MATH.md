# 数式とコードの対応

## 変更方針

元の `sample_path()`・`ResBlock`・`TinyUNet` はそのまま `ai/core.py` に分離。
元ノートは削除せず `notebooks/original.ipynb` に無変更で保存。
数学のアルゴリズムは変更していません。注意機構、最適輸送coupling、拡散のscore学習などは追加していません。
変更は中間状態の記録、再現用seed、学習再開、展示サーバー、可視化が中心です。
学習の標準設定は8→20epochに増やしましたが、品質向上が保証されたという意味ではありません。

## 1. 入力

MNISTの各画素pはToTensorで0〜1、Normalizeで x₁ = (p−0.5)/0.5 = 2p−1。
データは `(B,1,28,28)`。x₀ は同じ形の標準正規乱数。
ノイズを[-1,1]へ制限しない。最初から画素値として有効な範囲にある必要はありません。

## 2. 学習時の直線経路

\[
t\sim U[0,1],\quad x_0\sim\mathcal N(0,I),\quad x_1\sim p_{data},\quad x_0\perp x_1
\]
\[
x_t=(1-t)x_0+tx_1,\qquad \frac{dx_t}{dt}=x_1-x_0.
\]

`sample_path()`:

```python
x0 = torch.randn_like(x1)
t = torch.rand(x1.shape[0], device=x1.device)
t4 = t[:, None, None, None]
xt = (1 - t4) * x0 + t4 * x1
target_velocity = x1 - x0
```

各画像について時刻を1つサンプルし、全画素にbroadcastする。
微分は x₀,x₁ を固定してtについて行うので −x₀+x₁。

## 3. 速度を予測する

\[
v_\theta:\mathbb R^{784}\times[0,1]\to\mathbb R^{784}
\]

```python
predicted = model(xt, t)
```

TinyUNetは28→14→7→14→28の解像度で特徴を処理するCNN。
元のbase=32、time_dim=64、525,537パラメータ。
同じ解像度のencoder特徴をdecoderへ結合して細部を保持する。
時刻のsin/cos特徴は入力tの情報を各ResBlockへ渡すためのもの。
出力は画像と同じ形だが、**画像ではなく各画素の時間変化率**。最終層をsigmoid/tanhにせず、正負の速度を表現する。

## 4. MSE学習

\[
\mathcal L(\theta)=\mathbb E\left[\frac1{784}\|v_\theta(x_t,t)-(x_1-x_0)\|^2\right].
\]

```python
loss = F.mse_loss(predicted, target)
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

PyTorchのmeanはbatch・channel・画素で平均するので、期待値のミニバッチ近似と画素平均に対応。
同じ x,t でも複数の組(x₀,x₁)があり得る。MSEの最適予測は条件付き平均 E[x₁−x₀ | xₜ=x]。
モデルは特定の訓練ペアの直線を暗記して生成時に再生しているわけではない。

## 5. 生成時のODEとEuler法

\[
\frac{dx}{dt}=v_\theta(x,t),\quad x(0)\sim\mathcal N(0,I),
\quad \Delta t=1/N,\quad t_k=k/N.
\]
\[
x_{k+1}=x_k+\Delta t\,v_\theta(x_k,t_k).
\]

`ai/trajectory.py`:

```python
dt = 1.0 / steps
for k in range(steps):
    t = torch.full((count,), k / steps, device=device)
    v = model(x, t)
    x = x + dt * v
    velocities.append(v.cpu().clone())
    states.append(x.cpu().clone())
```

xの変化を、区間中は現在予測された速度で近似する1次Euler法。
滑らかさなどの条件下で大域誤差はO(1/N)。本アプリは元のEuler法を維持し、Heun/RKなどは使わない。
20/50/100stepの切替で計算精度と時間を比較できる。
小さい刻みでも、学習した速度場が不正確ならよい数字にはならない。

生成時に本物のx₁は使わない。各ステップで速度を予測し直すので、生成軌道は一般に直線ではない。
学習の直線補間をアニメーションとして表示することはしない。

## 6. 可視化

- メイン・最終結果: `clamp((x+1)/2,0,1)` を画面の明暗にする。計算状態は無加工。
- 速度: v>0を青緑（明るくする）、v<0を橙（暗くする）。強度は |v|/S。
- 変化量: |xₖ₊₁−xₖ| = |v|/N。最大値をHとして強度は |Δx|/H。
- Sは全時刻・全生成画像の|v|の最大値、H=S/N。再生途中で尺度は変えない。
- 次の画像: 保存された xₖ₊₁ を表示。描画による補間はしない。
- 最終時刻t=1には次の更新がないため、内部4枚は最後の区間を明示する。
- 同じseed・同じ生成枚数なら、ステップ数が違ってもCPUで作る初期ノイズは完全に同じ。
- 比較表示は時間スライダーに連動し、各経路の最寄りの離散時刻と正確な値を表示。

## 7. 品質改善の範囲

元実装に、時刻の向き・目標速度の符号・入力の正規化・生成時の復元の明らかな誤りは見つからなかった。
標準の学習率2e−4、batch128、base32を維持。学習を20epochまで継続可能にし、固定seedでの画像検証を用意した。
モデル拡大は `--base 48` / `--base 64` で可能だが、元のbase32をまず検証する。
既存checkpointへの構造変更はしない。自動学習率変更やEMAなど、理解する対象を増やす変更は加えていない。
実際にどの学習量まで試したか、改善を確認できたかはVERIFICATION.mdに記録する。
