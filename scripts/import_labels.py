#!/usr/bin/env python3
"""レビューツールが出力したラベルTSVを ground_truth.tsv に取り込む。

review.html の「エクスポート」で得た labels_review.tsv を、指定の教師データに
マージする(時刻が±1.5s以内の既存行は上書きせず保持、新規のみ追加、時刻順にソート)。

使い方: python3 import_labels.py <labels_review.tsv> [target_ground_truth.tsv]
"""
import os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read(path):
    rows=[]
    if os.path.exists(path):
        for l in open(path):
            if l.startswith("#") or not l.strip(): continue
            c=l.rstrip("\n").split("\t")
            if len(c)>=4: rows.append([float(c[1])]+c)
    return rows

def main(src,dst):
    cur=read(dst); new=read(src)
    have=[r[0] for r in cur]
    added=0
    for r in new:
        if any(abs(r[0]-t)<1.5 for t in have): continue
        cur.append(r); have.append(r[0]); added+=1
    cur.sort(key=lambda r:r[0])
    header=("# 笛の正解データ (ground truth)\n"
            "# mm:ss\tseconds\tf0(Hz)\tlabel\tverified\tnote\n")
    with open(dst,"w") as f:
        f.write(header)
        for r in cur: f.write("\t".join(r[1:])+"\n")
    print(f"{added}件追加 / 合計{len(cur)}件 -> {dst}")

if __name__=="__main__":
    if len(sys.argv)<2: print("usage: import_labels.py <labels_review.tsv> [target.tsv]"); sys.exit(1)
    src=sys.argv[1]
    dst=sys.argv[2] if len(sys.argv)>2 else os.path.join(ROOT,"labels","ground_truth.tsv")
    main(src,dst)
