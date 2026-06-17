#!/usr/bin/env python3
"""笛検出 (アンサンブル) — 本番用の候補生成器。

2つの専門家を併用して取りこぼしを減らす:
  detect_whistle_v1 : 低域(群衆)対比 + 長め持続。明瞭な笛に強い(高精度/再現は頭打ち)
  detect_whistle_v2 : 笛帯域の局所背景対比 + 短い持続。短い/弱い笛に強い(高再現)

両者の候補を合議スコア(各検出器での「上限−順位」を加算。両方に出れば加算され優先)で
グローバルにランク付けして返す。両方一致の候補ほど高信頼。出力は「検証すべき候補」で、
最終確定は人手確認が要る。確定した笛/誤検出は labels/ground_truth.tsv に蓄積する。

正解データ(labels/ground_truth.tsv)での動作点 (許容±4s, この試合=笛48個):
  N=15 → precision100% recall31% / N=30 → P73% R46%(F1最良) / N=120 → P36% R90%
Nを上げるほど再現率は上がるが誤検出が増える。用途に応じてNを選ぶ。

使い方:  python3 detect_whistle.py [input.wav] [N]   (既定 N=30)
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_whistle_v1 as v1
import detect_whistle_v2 as v2

POOL = 120   # 各専門家から取る候補数(ランク付けの母数)

def candidates(path):
    """全候補を合議スコア順(高い順)に返す: [(time, f0, score, sources), ...]"""
    v1r = sorted(v1.detect(path, POOL), key=lambda r: -r[1])   # (tt,pts,nd,ff)
    v2r = sorted(v2.detect(path, POOL), key=lambda r: -r[2])   # (tt,ff,pts,nd)
    N1, N2 = len(v1r), len(v2r)
    agg = {}   # key time -> [score, f0, sources]
    def add(t, f0, pts, src):
        k = next((k for k in agg if abs(k - t) < 2.0), None)
        if k is None: agg[t] = [pts, f0, {src}]
        else: agg[k][0] += pts; agg[k][2].add(src)
    for i, r in enumerate(v1r): add(r[0], r[3], N1 - i, "v1")
    for i, r in enumerate(v2r): add(r[0], r[1], N2 - i, "v2")
    rows = [(t, v[1], v[0], v[2]) for t, v in agg.items()]
    rows.sort(key=lambda r: -r[2])
    return rows

def detect(path, topn=30):
    return candidates(path)[:topn]

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "2026-l1-final-1st.wav")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print("# mm:ss\tseconds\tf0(Hz)\tscore\tsources")
    for t, f0, sc, src in detect(path, n):
        print(f"{int(t//60)}:{t%60:05.2f}\t{t:.2f}\t{f0:.0f}\t{sc}\t{'+'.join(sorted(src))}")
