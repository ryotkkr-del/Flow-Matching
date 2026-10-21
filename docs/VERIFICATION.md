# 動作確認結果（2026-10-21）

## 結論

Pythonの展示サーバー、AIの数式、保存経路、フロントエンドの状態管理を検証。
MNISTで4epochの実学習を実施し、学習済みcheckpointと9本の実生成経路を別添ZIPで提供する。
**本番品質の合格判定は未完了。ブラウザでの描画・操作、展示PC、長時間連続運転は未検証。**

GPUはなく、CPUで4epochまで学習して保存済みcheckpointから停止。
20epochの学習はこの実行では完了していない。Colabノートの再開手順で継続可能。
元ノートの8epoch結果を上回る生成品質が得られたという主張はしない。

## 学習の実測

- データ: MNIST training 60,000枚、各epochで全件使用。テスト画像による学習はしていない。
- 初期化: 自作TinyUNetを乱数初期化。外部の学習済み重みなし。
- base32 / time_dim64 / 525,537パラメータ、seed42、batch128、AdamW lr2e−4。
- torch.cuda.is_available() = False。
- 添付(3).ipynbは元のまま保存。sample_path・ResBlock・TinyUNetの内容一致をテスト。

| epoch | 学習MSE |
|---|---|
| 1 | 0.335590 |
| 2 | 0.230518 |
| 3 | 0.215621 |
| 4 | 0.209041 |

MSE低下は学習の進行を示すが、単独では生成品質の合格根拠にしない。
seed42/43/44の各4枚、50stepの合計12枚を目視確認。
1epoch時より背景のざらつきが減り、数字らしい形が増えたが、線の接続・形状が崩れたサンプルが残る。
FIDなどの定量評価、元ノートの8epochモデルとの同条件比較はしていない。
**この重みは試験用。20epochまで継続して、同じ固定seedで再確認することを推奨。**

## 正しさ・API・UIロジック

- `python -m pytest -q`: **9 passed**。
  元AIコードの保存、経路式、backward、Euler更新、clamp非適用、seed再現、checkpoint復元、
  非同期API、cache再利用、入力範囲、busy制御、例外、cache消失後の復旧を確認。
- `node --test tests/test_ui.cjs`: **6 passed**。
  経路検査、保存状態からのスライダー、停止と再開、最終時刻、色変換、比較時刻同期を確認。
  Node上のmock DOMを使用する単体テストであり、実ブラウザの描画確認とは別。
- `node --check web/app.js` / `python -m compileall -q ai server.py`: PASS。
- 実4epochモデルをFastAPI TestClientで生成し、応答を取得・Euler式を照合: PASS。
  4枚×50step、画像状態51点・速度50点、転送・JSON処理込み **0.832秒**。
  CPU環境依存の単発実測で、展示PCでの速度保証ではない。
- 元ノートSHA256: `216cf3cede1aa57626c11e78e11c644c16e759687ab6c0e47d836a4467ac41b1`。
- 配布checkpoint SHA256: `b34b003f767cfcfe453c9a5e38b0b81e5907041123a192e0cff7aea34596d0d5`。

## 保存デモ

seed42/43/44 × Euler20/50/100step × 各4枚。
すべて配布する同じ4epoch checkpointでCPU生成した実軌道。
各ファイルのSHA256対応・Euler整合・モデルからの全軌道再計算照合は `demo-verification.json` に記録。
任意のJSONが本物であると無条件に保証する仕組みではない。別の経路を持ち込む場合は `ai.verify` を実行する。

## 未検証と残作業

1. **生成品質**: 20epochまで継続学習し、固定seedのPNGを確認。必要なら30epochまで再開。
2. **実ブラウザ**: クラウドブラウザがアクセス制限でページを開けず、見た目・実クリック・全画面・ファイル選択の確認は未実施。
   代替ブラウザによる回避は行っていない。展示PCでREADMEの操作を一通り確認する。
3. **長時間**: 数時間〜終日の耐久運転は未実施。cache96MiB・ジョブ履歴64件の上限と単一workerは実装済み。
4. **オフライン**: 外部API・CDN・外部フォントを使わない構成。物理的な回線切断とWindows実機での起動確認は未実施。
5. **GPU / Colab**: GPU学習とDrive保存の手順は実装済みだが、この環境では未実行。

今回のStarletteはTestClientでhttpx非推奨の警告を1件出したが、テストは全件成功。
アプリの通常起動はTestClientを使わない。

最終書き出し検査で1本の圧縮ファイルの不完全保存を検出し、再生成した。書き出しは一時ファイル完了後に置換する方式へ変更し、最終9本すべての展開・Euler式・モデル再計算一致を確認した。
