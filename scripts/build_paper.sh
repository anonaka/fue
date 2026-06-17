#!/usr/bin/env bash
# 論文 docs/paper.tex / paper.pdf を docs/paper.md から生成する。
# 必要: pandoc, lualatex(TeX Live日本語), perl。図は docs/figs/*.pdf を参照。
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# タイトル/サブタイトル/著者/日付はメタデータ、本文は「要旨」以降を使う
{
  printf -- '---\n'
  printf 'title: "群衆雑音下のラグビー中継音声からのレフェリー笛検出"\n'
  printf 'subtitle: "― エネルギーベース手法の限界と音色ベース分類による打破 ―"\n'
  printf 'author: "野中哲"\n'
  printf 'date: "2026-06-17"\n'
  printf 'lang: ja\n'
  printf -- '---\n\n'
  awk '/^\*\*要旨\*\*/{p=1} p' docs/paper.md
} > /tmp/paper_meta.md

pandoc /tmp/paper_meta.md -s -o docs/paper.tex \
  -V documentclass=ltjsarticle -V geometry:margin=25mm -V linkcolor=blue

# 所属を \thanks(脚注)で付与 (pandocは生LaTeXを直接埋めにくいため後処理)
perl -i -pe 's/\\author\{野中哲\}/\\author{野中哲\\thanks{神奈川不惑クラブ}}/' docs/paper.tex

cd docs
lualatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
lualatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
rm -f paper.aux paper.log paper.out
echo "done: docs/paper.pdf  ($(ls -la paper.pdf | awk '{print $5}') bytes)"
