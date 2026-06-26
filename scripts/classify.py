#!/usr/bin/env python3
"""音色ベース分類器で候補を再ランクする。

アンサンブル候補に正解データで笛/非笛ラベルを付け、各候補の音色特徴を抽出して
ロジスティック回帰を学習。交差検証(out-of-fold)で正直に評価し、分類器スコアで
候補を再ランクしたP/R/F1を、従来の合議スコア順と比較する。
"""
import os, sys, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_whistle_v2 as V2, detect_whistle as D, evaluate as E

FPS=V2.FPS

def feats(mag,freqs,t):
    """時刻t周辺(±0.4s)から音色特徴ベクトルを抽出。"""
    c=int(t*FPS); w=int(0.4*FPS)
    a,b=max(0,c-w),min(mag.shape[0],c+w)
    seg=mag[a:b]                                   # (frames, bins)
    def band(lo,hi): return seg[:,(freqs>=lo)&(freqs<=hi)]
    wb=band(1700,2500); fb=freqs[(freqs>=1700)&(freqs<=2500)]
    pk=wb.max(1); pf=fb[wb.argmax(1)]
    f0=pf[pk.argmax()]
    bandsum=wb.sum(1)+1e-9
    e_band=band(1700,5000).sum()+1e-9
    e_low =band(150,1300).sum()+1e-9
    e_f0  =band(f0-120,f0+120).sum()
    e_2f0 =band(2*f0-200,2*f0+200).sum()
    e_3f0 =band(3*f0-250,3*f0+250).sum()
    e_high=band(5000,7000).sum()
    # 特徴
    prom = pk.max()/(np.median(pk)+1e-9)                 # 突出
    tonal= pk.max()/bandsum[pk.argmax()]                  # トーン集中度
    h_ratio=(e_f0+e_2f0)/e_band                           # 倍音が帯域に占める割合
    h2_1 = e_2f0/(e_f0+1e-9)                              # 第2倍音/基本波
    h3_1 = e_3f0/(e_f0+1e-9)
    low_ratio = e_low/e_band                              # 群衆(低域)/笛帯域
    high_ratio= e_high/e_band
    pitch_std = pf.std()                                  # ピッチ安定性(小さいほど笛)
    dur = (pk>pk.max()*0.5).sum()/FPS                     # 突出が続く長さ
    onset = np.diff(bandsum).max()/(bandsum.mean()+1e-9)  # 立ち上がり鋭さ
    flat = np.exp(np.log(wb.mean(0)+1e-9).mean())/(wb.mean(0).mean()+1e-9) # スペクトル平坦度(低いほどトーン)
    return np.array([prom,tonal,h_ratio,h2_1,h3_1,low_ratio,high_ratio,pitch_std,dur,onset,flat],float)

def logreg_cv(X,y,k=5,l2=1.0,it=2000,lr=0.3):
    n,d=X.shape; mu=X.mean(0); sd=X.std(0)+1e-9; Xs=(X-mu)/sd
    rng=np.random.default_rng(0); idx=rng.permutation(n); folds=np.array_split(idx,k)
    oof=np.zeros(n)
    for f in range(k):
        te=folds[f]; tr=np.concatenate([folds[j] for j in range(k) if j!=f])
        Xt=np.c_[np.ones(len(tr)),Xs[tr]]; yt=y[tr]
        w=np.zeros(d+1)
        for _ in range(it):
            p=1/(1+np.exp(-Xt@w)); g=Xt.T@(p-yt)/len(tr); g[1:]+=l2*w[1:]/len(tr)
            w-=lr*g
        Xe=np.c_[np.ones(len(te)),Xs[te]]; oof[te]=1/(1+np.exp(-Xe@w))
    return oof

def main(path, gt_path):
    gt=E.load_gt(gt_path)
    cand=D.candidates(path); ctimes=[c[0] for c in cand]
    print(f"候補 {len(cand)}個 / 正解 {len(gt)}個",file=sys.stderr)
    x=V2.load(path); mag,freqs=V2.spectrogram(x)
    X=np.array([feats(mag,freqs,t) for t in ctimes])
    y=np.array([1.0 if any(abs(t-g)<=E.TOL for g in gt) else 0.0 for t in ctimes])
    print(f"正例(候補中) {int(y.sum())} / 負例 {int((1-y).sum())}",file=sys.stderr)
    oof=logreg_cv(X,y)
    # 分類器スコアで再ランク
    order=np.argsort(-oof)
    ranked=[ctimes[i] for i in order]
    print("\n=== 分類器で再ランク (交差検証OOFスコア順) ===")
    print(f"{'N':>4}{'TP':>4}{'FP':>4}{'prec':>7}{'rec':>7}{'F1':>7}")
    best=(0,0)
    for N,tp,fp,fn,P,R,F in E.curve(ranked,gt):
        if F>best[1]: best=(N,F)
        if N in (15,20,30,40,50,70,100,120): print(f"{N:>4}{tp:>4}{fp:>4}{P*100:>6.0f}%{R*100:>6.0f}%{F*100:>6.0f}%")
    print(f"F1最良: N={best[0]} (F1={best[1]*100:.0f}%)")
    # 特徴の効き(全データ学習の係数)
    mu=X.mean(0);sd=X.std(0)+1e-9;Xs=np.c_[np.ones(len(X)),(X-mu)/sd];w=np.zeros(Xs.shape[1])
    for _ in range(3000):
        p=1/(1+np.exp(-Xs@w));g=Xs.T@(p-y)/len(y);g[1:]+=w[1:]/len(y);w-=0.3*g
    names=["prom","tonal","h_ratio","h2/1","h3/1","low_r","high_r","pitch_std","dur","onset","flat"]
    print("\n特徴の係数(正=笛らしさ↑):")
    for nm,c in sorted(zip(names,w[1:]),key=lambda z:-abs(z[1])): print(f"  {nm:10s} {c:+.2f}")

if __name__=="__main__":
    if len(sys.argv)<3: sys.exit("使い方: python3 classify.py <input.wav> <ground_truth.tsv>")
    main(sys.argv[1], sys.argv[2])
