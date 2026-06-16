#!/usr/bin/env python3
"""笛検出 (アンサンブル) — 本番用の候補生成器。

2つの専門家を併用して取りこぼしを減らす:
  detect_whistle_v1 : 低域(群衆)対比 + 長め持続。持続した明瞭な笛に強い(高精度)
  detect_whistle_v2 : 笛帯域の局所背景対比 + 短い持続。短い/群衆下の弱い笛に強い(高再現)

どちらも単独では不完全(v1は短い笛を、v2は持続笛の一部を取り逃す)。両者の和集合を
取ることで候補の再現率を上げる。ただし v2 は縦縞ノイズ等の誤検出も含むため、
出力は「検証すべき候補」であり、最終確定にはスペクトログラム等での人手確認が要る。
人手で確認した笛は labels/ground_truth.tsv に教師データとして蓄積する。

使い方:  python3 detect_whistle.py [input.wav] [N_each]
         各専門家から上位 N_each 件(既定25)を取り和集合を出力。
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_whistle_v1 as v1
import detect_whistle_v2 as v2

def detect(path, n_each=25):
    # v1: (tt,pts,nd,ff) / v2: (tt,ff,pts,nd) — 時刻=index0 で正規化
    items = [(r[0], r[3], "v1") for r in v1.detect(path, n_each)]      # (time, f0, src)
    items += [(r[0], r[1], "v2") for r in v2.detect(path, n_each)]
    items.sort()
    merged = []   # (time, f0, sources)
    for tt, ff, src in items:
        if merged and tt - merged[-1][0] < 2.0:
            merged[-1][2].add(src)
        else:
            merged.append([tt, ff, {src}])
    return merged

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "2026-l1-final-1st.wav")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    print("# mm:ss\tseconds\tf0(Hz)\tsources")
    for tt, ff, src in detect(path, n):
        print(f"{int(tt//60)}:{tt%60:05.2f}\t{tt:.2f}\t{ff:.0f}\t{'+'.join(sorted(src))}")
