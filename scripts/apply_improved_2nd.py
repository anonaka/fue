#!/usr/bin/env python3
"""改良版(長笛運用)を後半に適用して検出時刻を出力する。

手順:
  1. 前半候補(energy)で音色分類器を学習。
  2. 前半の out-of-fold 確率で「長い笛(>=0.4s)のF1」を最大化するしきい値 thr* を決める。
  3. 後半の候補(energy ∪ 長ridge) を分類器でスコアし、p>=thr* を検出として採用。
     (長ridge=倍音つき横リッジの持続>=0.4sのみ。マスクされた長笛を回収)
  4. 検出を results/whistles_2nd_improved.tsv に出力(make_chapters用)。
  5. 後半正解で precision/recall(長笛/全笛) を報告。

出力TSVを make_chapters.py に渡せばチャプター付き後半movが作れる。
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

def long_f1(times_kept, longs, all_gt):
    """長笛運用の P/R/F1。recall=長笛のみ, precision=何かの笛に当たった候補/採用数。"""
    if not len(times_kept): return 0,0,0
    hitany=sum(any(abs(t-g)<TOL for g in all_gt) for t in times_kept)
    rl=sum(any(abs(g-t)<TOL for t in times_kept) for g in longs)
    P=hitany/len(times_kept); R=rl/len(longs); F=2*P*R/(P+R) if P+R else 0
    return P,R,F

def main():
    w1=os.path.join(ROOT,"data","2026-l1-final-1st.wav")
    w2=os.path.join(ROOT,"data","2026-l1-final-2nd.wav")
    gt1=gt_whistles(os.path.join(ROOT,"labels","ground_truth.tsv"))
    gt2=gt_whistles(os.path.join(ROOT,"labels","ground_truth_2nd.tsv"))

    # --- 前半: 学習 + しきい値選定 ---
    print("前半 学習中...",file=sys.stderr)
    c1=D.candidates(w1); m1,f1=V2.spectrogram(V2.load(w1))
    X1=np.array([C.feats(m1,f1,t) for t,_,_,_ in c1])
    y1=np.array([1.0 if any(abs(t-g)<=TOL for g in gt1) else 0.0 for t,_,_,_ in c1])
    longs1=[g for g in gt1 if CLN.sustain(m1,f1,g)>=DUR_LONG]
    oof=C.logreg_cv(X1,y1)                      # 正直な確率
    t1=[t for t,_,_,_ in c1]
    bestthr=(0.5,0)
    for thr in np.linspace(0.1,0.9,33):
        kept=[t1[i] for i in range(len(t1)) if oof[i]>=thr]
        P,R,F=long_f1(kept,longs1,gt1)
        if F>bestthr[1]: bestthr=(thr,F)
    thr=bestthr[0]
    print("前半で選定した動作点: p>=%.2f (前半 長笛F1=%.0f%%)"%(thr,bestthr[1]*100))
    w,mu,sd=train_full(X1,y1)

    # --- 後半: 候補(energy ∪ 長ridge) をスコア → 検出 ---
    print("後半 適用中...",file=sys.stderr)
    c2=D.candidates(w2); m2,f2=V2.spectrogram(V2.load(w2))
    pool=[(t,f0) for t,f0,_,_ in c2]
    rid=CLN.ridge_candidates(m2,f2,topn=120)
    ridL=[(t,f) for t,f in rid if CLN.sustain(m2,f2,t,f)>=DUR_LONG]
    added=0
    for t,f in ridL:
        if not any(abs(t-pt)<2.0 for pt,_ in pool): pool.append((t,f)); added+=1
    print("後半候補: energy %d + 長ridge追加 %d = %d"%(len(c2),added,len(pool)))
    Xp=np.array([C.feats(m2,f2,t) for t,_ in pool])
    p=prob(Xp,w,mu,sd)
    det=sorted([(pool[i][0],pool[i][1],p[i]) for i in range(len(pool)) if p[i]>=thr])

    # --- 後半正解で評価 ---
    longs2=[g for g in gt2 if CLN.sustain(m2,f2,g)>=DUR_LONG]
    dt=[t for t,_,_ in det]
    P,R,F=long_f1(dt,longs2,gt2)
    rall=sum(any(abs(g-t)<TOL for t in dt) for g in gt2)
    print("\n=== 後半での検出結果(改良版・動作点 p>=%.2f) ==="%thr)
    print("検出 %d個  / 長笛recall %d/%d  全笛recall %d/%d"%(len(det),
          sum(any(abs(g-t)<TOL for t in dt) for g in longs2),len(longs2),rall,len(gt2)))
    print("長笛運用 P/R/F1 = %.0f%% / %.0f%% / %.0f%%"%(P*100,R*100,F*100))

    # --- TSV 出力 ---
    out=os.path.join(ROOT,"results","whistles_2nd_improved.tsv")
    with open(out,"w") as fo:
        fo.write("# 改良版(長笛運用)で後半に適用した検出結果。前半学習→p>=%.2f採用。\n"%thr)
        fo.write("# mm:ss\tseconds\tf0(Hz)\tlabel\tverified\tnote\n")
        for t,f0,pr in det:
            mm,ss=int(t//60),t%60
            fo.write("%d:%05.2f\t%.2f\t%.0f\twhistle\tauto\tp=%.2f\n"%(mm,ss,t,f0,pr))
    print("\n出力: %s (%d件)"%(out,len(det)))

if __name__=="__main__": main()
