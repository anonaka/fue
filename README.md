# fue — レフェリーの笛 検出

ラグビー試合の録画から、レフェリーの笛が吹かれた時刻を検出し、動画にチャプターマーカーとして埋め込む。

## フォルダ構成

```
data/                          サンプル入力データ
  2026-l1-final-1st.mov          元動画 (約3.5GB, h264/aac)
  2026-l1-final-1st.wav          抽出音声 (48kHz mono, 検出の入力)

labels/                        教師データ (ground truth)
  ground_truth.tsv               人手で確認した笛。将来の音色ベース検出器の学習・評価用

results/                       処理結果
  whistles.txt                   確定した笛24件(候補から人手確認したもの)
  chapters.ffmeta                ffmpeg用チャプターメタデータ
  2026-l1-final-1st_marked.mov   笛にチャプターマーカーを付けた動画(計24個)
  final_montage.png              スペクトログラム(目視確認用)

scripts/                       処理スクリプト
  detect_whistle.py              本番=アンサンブル(v1∪v2)。合議スコア順に候補を出力
  detect_whistle_v1.py           専門家1: 低域(群衆)対比+長め持続。明瞭な笛に強い(高精度)
  detect_whistle_v2.py           専門家2: 笛帯域の局所背景対比+短い持続。短い/弱い笛に強い(高再現)
  evaluate.py                    検出器を labels/ground_truth.tsv で評価(P/R/F1曲線)
  classify.py                    音色ベース分類器で候補を再ランク(交差検証で評価)
  make_review.py                 教師データ作成支援: 候補レビュー用HTMLを生成
  import_labels.py               レビュー結果のラベルTSVを ground_truth.tsv に取込
  preprocess.py                  HPSS前処理(実験用・未採用。NOTES.md参照)
  make_chapters.py               whistles.txt + 教師データ から marked.mov を生成
```

## セットアップ

依存パッケージは [uv](https://docs.astral.sh/uv/) で管理する。`uv run` が初回に仮想環境を
自動作成・同期するため、明示的なインストールは不要(各スクリプトを `uv run` で実行するだけ)。
音声の読み込みには別途 `ffmpeg` が必要。

```bash
uv sync   # 任意: 事前に .venv を作って依存を入れておく
```

## 使い方

```bash
# 1) 笛の候補を検出 (v1とv2の和集合)。sources列で各専門家の検出が分かる:
uv run scripts/detect_whistle.py > /tmp/candidates.tsv
#    sources=v1+v2 は両者一致で高信頼。v2単独は誤検出(縦縞ノイズ等)を含むので要確認。
#    確認のうえ results/whistles.txt を編集して確定リストにする。

# 2) whistles.txt + labels/ground_truth.tsv を元にマーカー付き動画を生成:
uv run scripts/make_chapters.py        # 前半
uv run scripts/make_chapters.py 2nd    # 後半(プリセット)
#   任意: make_chapters.py <src.mov> <out.mov> <labels.tsv>...
```

`make_chapters.py` は `results/whistles.txt`(確定リスト) と `labels/ground_truth.tsv`(人手確認の
教師データ)をマージし、時刻順にソート・近接重複(1秒以内)を除去してマーカーを作る。自動検出で
拾えない笛(下記)は教師データに追記すればマーカーに反映される(再検出で whistles.txt を上書きしても消えない)。

## 教師データ作成 (別試合のラベル付け)

全編を聴く代わりに、検出器の高再現な候補だけをレビューして効率的にラベル付けする:

```bash
# 1) 候補レビュー用HTMLを生成 (各候補に音声クリップ+スペクトログラムを埋込)
uv run scripts/make_review.py 別試合.wav            # -> results/review.html
open results/review.html
#    ブラウザでキーボード操作: W=笛 / N=非笛 / U=保留, Space=再生, ↑↓=移動
#    終わったら「エクスポート」で labels_review.tsv をダウンロード

# 2) ラベルを教師データに取り込む
uv run scripts/import_labels.py ~/Downloads/labels_review.tsv labels/別試合_gt.tsv
```

候補は再現率上限90%なので大半をカバーする。見逃し分(約10%)は教師データへ手動追記する。

## 検出方式

笛は約1.7〜2.5kHzの基本波＋倍音を持つ狭帯域トーン。観客の歓声(広帯域)と区別するため、
スペクトログラムを作り「基本波のトーン性・倍音の同時エネルギー・背景に対するコントラスト・持続」を
3指標で評価して合議する。2つの専門家は背景の取り方が異なる:

- **v1** = 笛 ÷ 低域(150-1300Hz)の群衆レベル。持続した明瞭な笛に強いが、短い笛や群衆が
  大きい場面の弱い笛を取り逃す。
- **v2** = 笛 ÷ 笛帯域そのものの局所背景(時間方向)。人の耳の周波数選択性(低域の群衆は
  2kHzの笛をマスクしない)を反映し、短い/群衆下の笛を拾えるが、縦縞ノイズ等の誤検出が増える。

両者は得意分野が相補的なので、本番は両者の候補を合議スコア順にランク付けして出す。

### 性能 (labels/ground_truth.tsv = 笛48個 で評価, 許容±4s)

`uv run scripts/evaluate.py` でP/R/F1曲線を出せる。上位N件を採るときの動作点:

| N | precision | recall | F1 |
|---|-----------|--------|-----|
| 15 | 100% | 31% | 48% |
| 30 | 73% | 46% | 56% (F1最良) |
| 70 | 44% | 65% | 53% |
| 120 | 36% | 90% | 51% |

- 上位15件は**誤検出ゼロ**。Nを上げると再現率は最大90%まで上がるが誤検出が急増。
- アンサンブルの**再現率上限は90%**(48個中43個)。残る4個(1:57, 11:35, 12:39, 23:57)は
  エネルギー/突出度ベースでは検出不能で、音色を学習した分類器が必要。

## 既知の限界 (なぜ人手の教師データが要るか)

検出はエネルギー/突出度ベースで、人の聴覚には及ばない。例えば **1:57 の笛**は、自分の
周波数帯の中で見ても背景の約1.7倍しか突出せず(検出可能な笛は3〜6倍)、周囲も2kHz帯が
ざわついているため、どの指標でも候補に上がらない。人の耳が明瞭に聞けるのは、エネルギー比
ではなく **音色・倍音パターンの認識、ピッチ周期性、聴覚的ストリーム分離、注意/予測** を
使っているため。これを自動化するには笛サンプルで学習した音色ベースの認識器が必要で、
それまでは `labels/ground_truth.tsv` に人手確認分を蓄積して補う。

## 必要環境

- Python 3 + numpy
- ffmpeg / ffprobe
