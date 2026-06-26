#!/usr/bin/env python3
"""分類器の汎化検証: 前半で学習し後半でテストする(別試合への転移)。

前半候補(正解で笛/非笛ラベル)で音色分類器を学習し、後半候補に適用。
後半(テスト)の正解で precision/recall/F1 を測り、合議スコア順(baseline)と比較する。

使い方: python3 classify_generalize.py <train.wav> <train_gt.tsv> <test.wav> <test_gt.tsv>
"""
import os,sys,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import detect_whistle as D, detect_whistle_v2 as V2, classify as CL, evaluate as E

def load_set(wav, gt_pos):
    cands=D.candidates(wav)                       # 合議スコア順
    mag,freqs=V2.spectrogram(V2.load(wav))
    X=np.array([CL.feats(mag,freqs,t) for t,_,_,_ in cands])
    y=np.array([1.0 if any(abs(t-g)<=E.TOL for g in gt_pos) else 0.0 for t,_,_,_ in cands])
    return cands,X,y

def gt_whistles(path):
    return [float(l.split("\t")[1]) for l in open(path)
            if not l.startswith("#") and l.strip() and l.split("\t")[3].strip()=="whistle"]

def train(X,y,l2=1.0,it=4000,lr=0.3):
    mu=X.mean(0);sd=X.std(0)+1e-9;Xs=np.c_[np.ones(len(X)),(X-mu)/sd];w=np.zeros(Xs.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-Xs@w));g=Xs.T@(p-y)/len(y);g[1:]+=l2*w[1:]/len(y);w-=lr*g
    return w,mu,sd
def score(X,w,mu,sd): return 1/(1+np.exp(-(np.c_[np.ones(len(X)),(X-mu)/sd]@w)))

def curve(order_times,gt,tag):
    flags,used=E.match(order_times,gt)
    print(f"[{tag}] 上限recall {sum(used)}/{len(gt)}")
    best=(0,0)
    for N,tp,fp,fn,P,R,F in E.curve(order_times,gt):
        if F>best[1]:best=(N,F,P,R,tp,fp)
    N,F,P,R,tp,fp=best
    print(f"  F1最良 N={N}: TP={tp} FP={fp} prec={P*100:.0f}% rec={R*100:.0f}% F1={F*100:.0f}%")
    return best

def main(w1, gt1_path, w2, gt2_path):
    gt1=gt_whistles(gt1_path)
    gt2=gt_whistles(gt2_path)
    print(f"前半 笛{len(gt1)} / 後半 笛{len(gt2)}",file=sys.stderr)
    print("前半(学習)読込...",file=sys.stderr); c1,X1,y1=load_set(w1,gt1)
    print("後半(テスト)読込...",file=sys.stderr); c2,X2,y2=load_set(w2,gt2)
    w,mu,sd=train(X1,y1)
    s2=score(X2,w,mu,sd)
    t2=[c[0] for c in c2]
    print("\n=== 後半(2nd half)での評価 ===")
    curve(t2,gt2,"baseline 合議スコア順")
    order=[t2[i] for i in np.argsort(-s2)]
    curve(order,gt2,"分類器(前半で学習→後半に適用)")

if __name__=="__main__":
    if len(sys.argv)<5: sys.exit("使い方: python3 classify_generalize.py <train.wav> <train_gt.tsv> <test.wav> <test_gt.tsv>")
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
