#!/usr/bin/env python3
"""改良版(長笛運用)を任意の音声/動画に適用して笛を検出する。

前半(2026 L1 final 1st)で音色分類器を学習し、長い笛(>=0.4s)のF1を最大化する
しきい値を選定。対象音声の候補(energy ∪ 長ridge)をスコアし p>=thr を検出として出力。
正解が無い新規試合への適用を想定(評価はせず検出のみ出力)。

使い方: python3 apply_improved.py <in.wav|in.mov> [out.tsv]
"""
import os, sys, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_whistle as D, detect_whistle_v2 as V2, evaluate as E
import classify as C, classify_long as CLN

TOL=E.TOL; DUR_LONG=CLN.DUR_LONG

def gt_whistles(p):
    return [float(l.split("\t")[1]) for l in open(p)
            if not l.startswith("#") and l.strip() and l.split("\t")[3].strip()=="whistle"]
def train_full(X,y,l2=1.0,it=4000,lr=0.3):
    mu=X.mean(0); sd=X.std(0)+1e-9; Xs=np.c_[np.ones(len(X)),(X-mu)/sd]; w=np.zeros(Xs.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-Xs@w)); g=Xs.T@(p-y)/len(y); g[1:]+=l2*w[1:]/len(y); w-=lr*g
    return w,mu,sd
def prob(X,w,mu,sd): return 1/(1+np.exp(-(np.c_[np.ones(len(X)),(X-mu)/sd]@w)))

def select_threshold(X1,y1,t1,gt1,longs1):
    oof=C.logreg_cv(X1,y1); best=(0.5,0)
    for thr in np.linspace(0.1,0.9,33):
        kept=[t1[i] for i in range(len(t1)) if oof[i]>=thr]
        if not kept: continue
        hitany=sum(any(abs(t-g)<TOL for g in gt1) for t in kept)
        rl=sum(any(abs(g-t)<TOL for t in kept) for g in longs1)
        P=hitany/len(kept); R=rl/len(longs1); F=2*P*R/(P+R) if P+R else 0
        if F>best[1]: best=(thr,F)
    return best

def main(src, train_wav, train_gt, out=None):
    w1=train_wav
    gt1=gt_whistles(train_gt)
    print("前半 学習中...",file=sys.stderr)
    c1=D.candidates(w1); m1,f1=V2.spectrogram(V2.load(w1))
    X1=np.array([C.feats(m1,f1,t) for t,_,_,_ in c1])
    y1=np.array([1.0 if any(abs(t-g)<=TOL for g in gt1) else 0.0 for t,_,_,_ in c1])
    t1=[t for t,_,_,_ in c1]; longs1=[g for g in gt1 if CLN.sustain(m1,f1,g)>=DUR_LONG]
    thr,f1score=select_threshold(X1,y1,t1,gt1,longs1)
    print("動作点 p>=%.2f (前半 長笛F1=%.0f%%)"%(thr,f1score*100))
    w,mu,sd=train_full(X1,y1)

    print("対象 候補生成中: %s"%src,file=sys.stderr)
    c2=D.candidates(src); m2,f2=V2.spectrogram(V2.load(src))
    pool=[(t,f0) for t,f0,_,_ in c2]
    rid=CLN.ridge_candidates(m2,f2,topn=120)
    ridL=[(t,f) for t,f in rid if CLN.sustain(m2,f2,t,f)>=DUR_LONG]
    added=0
    for t,f in ridL:
        if not any(abs(t-pt)<2.0 for pt,_ in pool): pool.append((t,f)); added+=1
    print("候補: energy %d + 長ridge追加 %d = %d"%(len(c2),added,len(pool)))
    Xp=np.array([C.feats(m2,f2,t) for t,_ in pool])
    p=prob(Xp,w,mu,sd)
    det=sorted([(pool[i][0],pool[i][1],p[i]) for i in range(len(pool)) if p[i]>=thr])
    longd=sum(1 for t,f,_ in det if CLN.sustain(m2,f2,t,f)>=DUR_LONG)
    f0s=np.array([f for _,f,_ in det])
    print("\n=== 検出 %d個 (うち長笛 %d) ==="%(len(det),longd))
    if len(det):
        print("f0(Hz): med %.0f / 範囲 %.0f-%.0f"%(np.median(f0s),f0s.min(),f0s.max()))

    if out is None:
        base=os.path.splitext(os.path.basename(src))[0]
        out=os.path.join(os.path.dirname(src),"whistles_%s.tsv"%base)
    with open(out,"w") as fo:
        fo.write("# 改良版(長笛運用)を適用した自動検出。学習=%s, p>=%.2f。正解なし。\n"%(os.path.basename(train_wav),thr))
        fo.write("# mm:ss\tseconds\tf0(Hz)\tlabel\tverified\tnote\n")
        for t,f0,pr in det:
            mm,ss=int(t//60),t%60
            fo.write("%d:%05.2f\t%.2f\t%.0f\twhistle\tauto\tp=%.2f\n"%(mm,ss,t,f0,pr))
    print("出力: %s (%d件)"%(out,len(det)))

if __name__=="__main__":
    if len(sys.argv)<4: sys.exit("使い方: python3 apply_improved.py <apply.wav|mov> <train.wav> <train_gt.tsv> [out.tsv]")
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv)>4 else None)
