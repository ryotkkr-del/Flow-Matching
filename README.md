# FLOW — ノイズが、かたちになる。

白鷺祭 / OIF向け、MNIST Flow Matchingのオフライン展示アプリ。
28×28のノイズが数字になる実際のODE軌道を、再生・停止・スライダーで観察します。

**実装済みのアプリです。ただし学習・品質確認の完了状況は [動作確認結果](docs/VERIFICATION.md) を確認してください。**
GitHubのソースだけには学習済み重みを同梱していません。自分で学習するか、別添の実学習済みデータを配置してください。
学習済みモデルもデモもない場合、架空の生成画像は表示せず、配置案内を出します。

## 1. Windowsで準備（インターネットが必要なのはここまで）

Python 3.11 / 3.12を推奨。GitHubのCode → Download ZIPでも入手可能。
PowerShellで展開先に移動して実行します。

```powershell
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` は対応バージョン範囲です。今回検証した実バージョンは `docs/environment.txt`。
GPU学習はColabの既存PyTorch環境を使うのが簡単です。

## 2. MNISTで学習する

おすすめは[Colabで学習ノートを開く](https://colab.research.google.com/github/ryotkkr-del/Flow-Matching/blob/main/notebooks/Train_Colab.ipynb)方法です。
学習は元のノートブックと同じ、無条件Flow Matchingです。外部の学習済みモデルは使いません。
自分のPCでも次のように実行できますが、CPUでは長時間かかります。

```powershell
.venv\Scripts\python -m ai.train --epochs 20 --batch-size 128 --lr 0.0002
```

初回はMNISTを `data/` に取得。学習中は各epoch終了時に `checkpoints/mnist.pt` を安全に更新します。
`--epochs` は追加回数でなく合計の目標epochです。再開時の学習率は、`--lr` を省略すると保存値を維持し、指定するとその値へ変更します。

```powershell
.venv\Scripts\python -m ai.train --resume checkpoints/mnist.pt --epochs 20 --offline
```

元のノートブックで作った `flow_matching_mnist_checkpoint.pt` も、標準のbase=32なら再利用可能です。
`checkpoints/mnist.pt` にコピーするか、`--resume` に元のファイルを指定します。
元ノートのcheckpointには乱数状態がないため、同じ学習経路の厳密な再現まではできません。
`--max-batches` は動作確認用で、そのモデルは展示・デモ出力を拒否します。

学習lossの低下だけで品質を合格にしないでください。次の出力で数字を目視確認します。

```powershell
.venv\Scripts\python -m ai.export --checkpoint checkpoints/mnist.pt --seeds 42 43 44 --steps 20 50 100 --count 4
```

`demos/` に実際の全中間状態・速度（`.json.gz`）と最終画像（`.png`）を保存します。
20epochは品質を保証する数値ではありません。まず固定seedで12枚を確認し、必要なら20→30epochへ継続します。
モデルを大きくする前に学習時間と生成出力を確認してください。

## 3. 展示を起動（以降はネット接続不要）

`checkpoints/mnist.pt` を配置後、`start.bat` をダブルクリック。
ブラウザで **http://127.0.0.1:8000** を開きます。終了は起動した画面で `Ctrl+C`。
黒い起動画面を閉じずに使ってください。全画面ボタンで展示表示にできます。

```powershell
.venv\Scripts\python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

macOS / Linuxでは `.venv/bin/python` で同様に環境を作り、`./start.sh`。
CDN、外部フォント、外部APIは使用しません。起動時に学習やMNIST取得はしません。

### 保存デモだけで展示する

チェックポイントなしでも、`demos/*.json.gz` があればその軌道を自動再生します。
Python環境と依存パッケージは必要です。新規生成をしないPCでは `pip install -r requirements-replay.txt` だけで再生可能です（PyTorch不要）。画面下部の「経路を読み込む」からJSON/gzの手動読込も可能。同じseed・異なるステップ数の保存デモがあれば、新規推論なしで比較できます。枚数は保存デモと合わせてください。
チェックポイントなしでは新しい生成・新しいステップ数での推論はできません。
本番前に複数seed・ステップ数の軌道を保存しておくとCPUの待ち時間を避けられます。

### 操作

- 再生・停止・再開、最初に戻る、時間スライダー、0.5× / 1× / 2×。
- ノイズ番号を指定して生成。「別のノイズへ」でランダムな初期状態に変更。
- 1枚 / 4枚を同時生成。下の画像を選ぶと、その画像の内部計算を拡大。
- ステップ数を変えて「同じノイズ・別のステップで比較」。番号・枚数・モデルが同じ場合だけ比較。
- 現在の画像、最終画像、予測速度、次の画像、変化量を並列表示。
- 「自動展示」で、保存した実生成デモを巡回。50stepのデモを優先し、なければ他のデモを使います。保存デモがなくても現在の軌道を繰り返し再生できます。
- 再生ボタン・スライダー・デモ選択で自動展示を終了。途中で届いた自動読込結果が手動操作を上書きしないよう制御しています。
- 経路保存・再読込。スライダーは保存した実計算値を使い、推論し直しません。
- 最終時刻の内部表示は最後の更新区間。最終時刻に架空の速度は追加しません。

### 復旧

画面が接続エラーになったらサーバーの起動画面で例外を確認し、Ctrl+C → start.bat。
GPU不要。CPUは4threadが標準。重い場合は生成を1枚・20stepへ。
`FLOW_THREADS` 環境変数でCPU thread数、`FLOW_CHECKPOINT` でモデルのパスを変更できます。
一度に1件だけ生成し、2件目は429で断ります。生成は別threadで実行。
軌道キャッシュは圧縮後96MiBまで、ジョブ履歴64件まで。ページを再読込してもサーバー内cacheは利用できます。
サーバー再起動後も使うには経路ファイルを保存してください。

## 構成

| ファイル | 役割 |
|---|---|
| `notebooks/original.ipynb` | 添付(3).ipynbの無変更コピー |
| `ai/core.py` | 元ノートのsample_path・ResBlock・TinyUNetを維持 |
| `ai/train.py` | MSE学習・再開・checkpoint保存 |
| `ai/trajectory.py` | Euler積分と全状態・速度の取得 |
| `ai/export.py` | 実モデルの軌道・最終画像を書き出す |
| `ai/verify.py` | checkpointから軌道を再計算して照合 |
| `server.py` | ローカルHTTP・非同期推論・上限付きcache |
| `web/` | ダークテーマ、canvas可視化、再生UI |
| `docs/MATH.md` | 数式とPythonの対応、元実装の検証 |
| `docs/VERIFICATION.md` | 実施した検証と未確認事項 |

## データの出所確認

```powershell
.venv\Scripts\python -m ai.verify demos/seed42-steps50.json.gz --checkpoint checkpoints/mnist.pt
```

ファイルに記録されたSHA256がcheckpointと一致し、同じモデルで再計算した軌道と照合できることを確認。
JSONにhashが書かれているだけでは出所の証明にはなりません。画面の手動読込時はEuler整合性を検査しますが、モデルの再計算までは行いません。
実際のモデルの出力だけを展示に使ってください。

## 開発用検証

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
```

Node.jsがある場合 `node --check web/app.js` と `node --test tests/test_ui.cjs`。
UIのロジック単体テストはブラウザの見た目・操作確認とは別です。
`npm run dev` はUIレビュー用の保存デモ専用サーバーで、新規推論はありません。展示の本体はPythonサーバーです。

## 任意のCPU学習高速化

対応CPUでは `--bf16` を付けると、一部の中間演算にbfloat16を使えます。重み・optimizer・MSEはfloat32。
確率経路、TinyUNetの構造、予測する速度、学習目標は同じです。丸め誤差があるため同一の学習経路になる保証はありません。
標準は従来のfloat32のまま。展示時の推論・経路書き出しはfloat32で行います。

```powershell
.venv\Scripts\python -m ai.train --resume checkpoints/mnist.pt --epochs 20 --threads 8 --bf16
```

古いCPUでは速くならない場合があります。今回の実測はVERIFICATION.mdに記録します。
`--stop-file STOP_TRAINING` を指定しておくと、アプリ直下に `STOP_TRAINING` ファイルを作ったとき、次のepoch完了・保存後に停止します。再開前にそのファイルを削除してください。

## 学習前後の比較を再現する

```powershell
.venv\Scripts\python -m ai.evaluate --before checkpoints/exhibit.pt --after checkpoints/mnist20.pt
```

MNISTのtest先頭1024枚を使ってFlow Matchingの損失を同じノイズ・時刻で比較し、同じ初期ノイズから64枚ずつ生成します。
評価時に学習は行いません。`quality/` にbefore/afterのPNG、対応する比較画像、損失・checkpoint識別子を保存。
評価損失は生成画像の品質スコアではありません。PNGの目視確認も必要です。

実モデルのAPIを繰り返し確認する場合は、`python tools/soak.py --seconds 900 --interval 10`。
これはASGIサーバーの検査です。ブラウザの描画、物理的な回線切断、Windows実機での確認を代替しません。
