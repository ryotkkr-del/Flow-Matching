# 数式とコードの対応

## 変更方針

元の `sample_path()`・`ResBlock`・`TinyUNet` はそのまま `ai/core.py` に分離。
元ノートは削除せず `notebooks/original.ipynb` に無変更で保存。
Flow Matchingの確率経路・速度予測・MSE・Euler更新は変更していません。任意の重み平均（EMA）の追加は第9節で説明します。注意機構、最適輸送coupling、拡散のscore学習などは追加していません。
変更は中間状態の記録、再現用seed、学習再開、展示サーバー、可視化が中心です。
学習の標準設定は8→20epochに増やしましたが、品質向上が保証されたという意味ではありません。

## 1. 入力

MNISTの各画素pはToTensorで0〜1、Normalizeで x₁ = (p−0.5)/0.5 = 2p−1。
データは `(B,1,28,28)`。x₀ は同じ形の標準正規乱数。
ノイズを[-1,1]へ制限しない。最初から画素値として有効な範囲にある必要はありません。

## 2. 学習時の直線経路

$$
t\sim U[0,1],\quad x_0\sim\mathcal N(0,I),\quad x_1\sim p_{data},\quad x_0\perp x_1
$$
$$
x_t=(1-t)x_0+tx_1,\qquad \frac{dx_t}{dt}=x_1-x_0.
$$

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

$$
v_\theta:\mathbb R^{784}\times[0,1]\to\mathbb R^{784}
$$

```python
predicted = model(xt, t)
```

TinyUNetは28→14→7→14→28の解像度で特徴を処理するCNN。
元のbase=32、time_dim=64、525,537パラメータ。
同じ解像度のencoder特徴をdecoderへ結合して細部を保持する。
時刻のsin/cos特徴は入力tの情報を各ResBlockへ渡すためのもの。
出力は画像と同じ形だが、**画像ではなく各画素の時間変化率**。最終層をsigmoid/tanhにせず、正負の速度を表現する。

## 4. MSE学習

$$
\mathcal L(\theta)=\mathbb E\left[\frac1{784}\|v_\theta(x_t,t)-(x_1-x_0)\|^2\right].
$$

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

$$
\frac{dx}{dt}=v_\theta(x,t),\quad x(0)\sim\mathcal N(0,I),
\quad \Delta t=1/N,\quad t_k=k/N.
$$
$$
x_{k+1}=x_k+\Delta t\,v_\theta(x_k,t_k).
$$

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
既存checkpointへの構造変更はしない。自動学習率変更は加えていない。EMAは任意の重み平均として第9節に分離し、使用しない基本実装も維持する。
実際にどの学習量まで試したか、改善を確認できたかはVERIFICATION.mdに記録する。

## 8. 追加学習時の任意の高速化

`--bf16` は確率経路・モデルの構造・MSE・最適化手順を変えず、一部の中間演算の数値形式を変えるだけのオプション。
bfloat16の丸め誤差が入るため、float32と完全に同じ結果になるわけではない。
パラメータ、optimizer状態、MSE計算はfloat32を維持し、書き出しと展示の推論はfloat32。
初期checkpointの固定入力では、損失の相対差が約0.11％で、全勾配が有限値であることを確認してから追加学習に使用した。
標準コマンドはfloat32のままで、元のコードとの対応を保つ。

checkpointを読み込む際は、読み込んだ同じbytesからSHA256を計算する。
学習側のファイル置換と起動が重なっても、別の重みの識別子が付かないようにするため。

## 9. 任意の重み移動平均

学習で更新する通常のパラメータを θₖ、平均用に別に保持するパラメータを θ̄ₖ とする。
AdamWで通常の重みを更新した後、

$$
\bar\theta_{k+1}=\beta\bar\theta_k+(1-\beta)\theta_{k+1},\qquad \beta=0.995.
$$

```python
for averaged, current in zip(ema.parameters(), model.parameters()):
    averaged.mul_(ema_decay).add_(current, alpha=1-ema_decay)
```

配列として見れば各成分の加重平均。新しく作るときの θ̄ は、その時点の θ のコピー。
重みの更新ごとの揺れを減らし、推論用の速度場を安定させることが目的。効果は比較して確認する。
EMAを使う生成でも、

$$
\frac{dx}{dt}=v_{\bar\theta}(x,t),\qquad x_{k+1}=x_k+\Delta t\,v_{\bar\theta}(x_k,t_k)
$$

という同じオイラー法であり、画像そのものを平均・補間しているわけではない。
学習用のMSEとoptimizerは通常の θ を更新し続ける。EMAの重みには勾配を流さない。
checkpointは通常のmodel・optimizerとEMAを別々に保存するため、再開時に平均重みと通常のoptimizerが混ざらない。

`load_model` はEMAがあれば標準でそれを選び、`use_ema=False` で通常の重みを選ぶ。
同じcheckpoint内に2種類の重みがあるため、生成経路にはSHA256とともに `weights: raw/ema` を記録する。
照合時もその種類を選択し、ステップ数だけの比較では種類が異なる経路を拒否する。

## 10. TinyUNetの中身をコード順に追う

対応する実装は [ai/core.py](../ai/core.py)。ニューラルネットの入力は現在の画像と時刻、出力は784画素それぞれの速度。

### 時刻を数値特徴に変える

`time_features(t)` の16個の周波数は、i=0,…,15について

$$
f_i=\exp\!\left(\frac{i}{15}\log100\right)=100^{i/15}.
$$

時刻tを32個の値へ変換する。

$$
\phi(t)=[\sin(2\pi f_0t),\ldots,\sin(2\pi f_{15}t),
\cos(2\pi f_0t),\ldots,\cos(2\pi f_{15}t)].
$$

この特徴を2層の線形変換に通すのが `time_mlp`。

$$
e(t)=W_b\,\operatorname{SiLU}(W_a\phi(t)+b_a)+b_b,
\qquad \operatorname{SiLU}(z)=\frac{z}{1+e^{-z}}.
$$

学習時、個々の直線の速度は一定でも、同じ途中画像から推測すべき速度の平均は時刻で変わる。
そのためモデルは画像だけでなくtも受け取る。

### ResBlockの処理

`*` を畳み込み、GNをGroupNorm、Sをチャンネル数調整用の1×1畳み込み（調整不要なら恒等写像）とする。
コードの処理順は次のとおり。

$$
h_1=\operatorname{SiLU}(\operatorname{GN}_1(W_1*x+b_1)),
$$
$$
h_2=h_1+(Ae(t)+c)_{:,:,\mathrm{None},\mathrm{None}},
$$
$$
y=\frac{S(x)+\operatorname{GN}_2(W_2*\operatorname{SiLU}(h_2)+b_2)}{\sqrt2}.
$$

時刻のベクトルを各画素に同じように加える部分が `[:, :, None, None]`。
`/sqrt(2)` は2つの枝を加えたときの大きさの増加を抑える固定係数で、分散が厳密に一定になるという意味ではない。
このブロック内の特徴の加算は、ODEで画像を更新するEuler法とは別の処理。

GroupNormは各画像のチャンネルを8群に分け、各群のチャンネル・縦・横にわたって平均μと分散σ²を計算する。
各要素の正規化は

$$
\operatorname{GN}(h)_i=\gamma_i\frac{h_i-\mu_g}{\sqrt{\sigma_g^2+\varepsilon}}+\beta_i.
$$

平均・分散の母数はその群の要素数。γとβはチャンネルごとに学習し、空間方向へbroadcastする。
他の画像の統計量は使わないため、1枚だけでも同じ仕組みで動く。

### 解像度とチャンネルの変化

標準の `base=32` の場合。

| 処理 | 空間サイズ | 出力チャンネル | コード上の操作 |
|---|---|---|---|
| enc1 | 28×28 | 32 | ResBlock(1,32) |
| enc2 | 14×14 | 64 | 2×2平均プーリング→ResBlock(32,64) |
| mid | 7×7 | 128 | 2×2平均プーリング→ResBlock(64,128) |
| dec2 | 14×14 | 64 | 最近傍拡大→enc2と結合(128+64)→ResBlock(192,64) |
| dec1 | 28×28 | 32 | 最近傍拡大→enc1と結合(64+32)→ResBlock(96,32) |
| output | 28×28 | 1 | SiLU→3×3畳み込み |

平均プーリングは2×2の4個の値を平均するだけ。最近傍拡大は元の値を2×2へ複製するだけで、そこに学習パラメータはない。
`torch.cat(...,dim=1)` は同じ位置の特徴をチャンネル方向に並べる。
最終出力にsigmoid/tanhを付けないのは、速度が正にも負にもなり、[-1,1]の範囲にも制限されないため。
