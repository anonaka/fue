#!/usr/bin/env python3
"""検出器を正解データ(labels/ground_truth.tsv)で評価する。

候補リスト(時刻,スコア順)を受け取り、上位Nを増やしながら
TP/FP/FN・precision/recall/F1 を計算する。
"""
import os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT=os.path.join(ROOT,"labels","ground_truth.tsv")
TOL=4.0

def load_gt():
    pos=[]
    for l in open(GT):
        if l.startswith("#") or not l.strip(): continue
        c=l.split("\t")
        if len(c)>=4 and c[3].strip()=="whistle": pos.append(float(c[1]))
    return sorted(pos)

def match(cand_times, gt):
    """cand_times: ランク順(高い順)の検出時刻リスト。貪欲に1対1照合。"""
    used=[False]*len(gt); tp=0; tp_flags=[]
    for d in cand_times:
        hit=-1; best=TOL+1
        for i,g in enumerate(gt):
            if not used[i] and abs(g-d)<best: best=abs(g-d); hit=i
        if hit>=0 and best<=TOL: used[hit]=True; tp+=1; tp_flags.append(True)
        else: tp_flags.append(False)
    return tp_flags, used

def curve(cand_times, gt, Ns=None):
    flags,_=match(cand_times,gt)
    G=len(gt)
    if Ns is None: Ns=list(range(5,len(cand_times)+1,5))+[len(cand_times)]
    rows=[]
    for N in Ns:
        tp=sum(flags[:N]); fp=N-tp; fn=G-tp
        P=tp/N if N else 0; R=tp/G; F=2*P*R/(P+R) if P+R else 0
        rows.append((N,tp,fp,fn,P,R,F))
    return rows

def main(path=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import detect_whistle as D
    if path is None: path=os.path.join(ROOT,"data","2026-l1-final-1st.wav")
    gt=load_gt()
    cand=D.candidates(path); times=[c[0] for c in cand]
    _,used=match(times,gt)
    print(f"正解 {len(gt)}個 / 候補 {len(cand)}個 / 上限recall {sum(used)}/{len(gt)}={sum(used)/len(gt)*100:.0f}%  (許容±{TOL}s)")
    print(f"{'N':>4}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>7}{'rec':>7}{'F1':>7}")
    bestF=(0,0)
    for N,tp,fp,fn,P,R,F in curve(times,gt):
        if F>bestF[1]: bestF=(N,F)
        print(f"{N:>4}{tp:>4}{fp:>4}{fn:>4}{P*100:>6.0f}%{R*100:>6.0f}%{F*100:>6.0f}%")
    print(f"F1最良: N={bestF[0]} (F1={bestF[1]*100:.0f}%)")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else None)
