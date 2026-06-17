#!/usr/bin/env python3
"""論文用の図を生成する (docs/figs/*.png/.pdf)。"""
import os,sys,numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import detect_whistle as D, detect_whistle_v2 as V2, classify as CL, evaluate as E
import classify_generalize as CG

W1=os.path.join(ROOT,"data","2026-l1-final-1st.wav")
W2=os.path.join(ROOT,"data","2026-l1-final-2nd.wav")
FIG=os.path.join(ROOT,"docs","figs")
FEAT=["prom","tonal","h_ratio","h2/1","h3/1","low_r","high_r","pitch_std","dur","onset","flat"]

def full_curve(order_times,gt):
    flags,_=E.match(order_times,gt); G=len(gt); Ns=range(1,len(order_times)+1)
    P=[];R=[];F=[]
    for N in Ns:
        tp=sum(flags[:N]); p=tp/N; r=tp/G; f=2*p*r/(p+r) if p+r else 0
        P.append(p);R.append(r);F.append(f)
    return np.array(list(Ns)),np.array(P),np.array(R),np.array(F)

def save(fig,name):
    for ext in ("png","pdf"): fig.savefig(os.path.join(FIG,name+"."+ext),bbox_inches="tight",dpi=150)
    plt.close(fig); print("  ",name)

def main():
    gt1=CG.gt_whistles(os.path.join(ROOT,"labels","ground_truth.tsv"))
    gt2=CG.gt_whistles(os.path.join(ROOT,"labels","ground_truth_2nd.tsv"))
    print("前半読込...",file=sys.stderr); c1,X1,y1=CG.load_set(W1,gt1)
    print("後半読込...",file=sys.stderr); c2,X2,y2=CG.load_set(W2,gt2)
    t1=[c[0] for c in c1]; t2=[c[0] for c in c2]
    oof=CL.logreg_cv(X1,y1)                 # 前半 交差検証スコア
    w,mu,sd=CG.train(X1,y1)                  # 前半 全データ学習(汎化用・係数用)
    s2=CG.score(X2,w,mu,sd)

    # 並び: baseline=合議スコア順(候補のまま), 分類器=スコア降順
    b1=full_curve(t1,gt1); k1=full_curve([t1[i] for i in np.argsort(-oof)],gt1)
    b2=full_curve(t2,gt2); k2=full_curve([t2[i] for i in np.argsort(-s2)],gt2)

    # Fig1: F1 vs N (1st half)
    fig,ax=plt.subplots(figsize=(5,3.4))
    ax.plot(b1[0],b1[3],label="baseline (consensus)",color="#888",lw=2)
    ax.plot(k1[0],k1[3],label="timbre classifier (CV)",color="#1565c0",lw=2)
    ax.set_xlabel("N (top candidates)");ax.set_ylabel("F1");ax.set_title("1st half: F1 vs N")
    ax.grid(alpha=.3);ax.legend();ax.set_ylim(0,1)
    save(fig,"fig1_f1_1st")

    # Fig2: PR curve (1st half)
    fig,ax=plt.subplots(figsize=(5,3.4))
    ax.plot(b1[2],b1[1],label="baseline",color="#888",lw=2)
    ax.plot(k1[2],k1[1],label="classifier (CV)",color="#1565c0",lw=2)
    ax.set_xlabel("recall");ax.set_ylabel("precision");ax.set_title("1st half: Precision-Recall")
    ax.grid(alpha=.3);ax.legend();ax.set_xlim(0,1);ax.set_ylim(0,1.02)
    save(fig,"fig2_pr_1st")

    # Fig3: generalization F1 vs N (2nd half)
    fig,ax=plt.subplots(figsize=(5,3.4))
    ax.plot(b2[0],b2[3],label="baseline (consensus)",color="#888",lw=2)
    ax.plot(k2[0],k2[3],label="classifier (trained on 1st)",color="#c62828",lw=2)
    ax.set_xlabel("N (top candidates)");ax.set_ylabel("F1");ax.set_title("2nd half (held-out): F1 vs N")
    ax.grid(alpha=.3);ax.legend();ax.set_ylim(0,1)
    save(fig,"fig3_generalize")

    # Fig4: feature coefficients
    coef=w[1:]; order=np.argsort(coef)
    fig,ax=plt.subplots(figsize=(5,3.6))
    cols=["#c62828" if coef[i]<0 else "#1565c0" for i in order]
    ax.barh([FEAT[i] for i in order],[coef[i] for i in order],color=cols)
    ax.axvline(0,color="k",lw=.6);ax.set_xlabel("logistic regression coefficient (standardized)")
    ax.set_title("Feature importance (+ = whistle-like)");ax.grid(axis="x",alpha=.3)
    save(fig,"fig4_coef")

    print(f"\nbaseline 1st F1max={b1[3].max():.2f} / classifier 1st F1max={k1[3].max():.2f}",file=sys.stderr)
    print(f"baseline 2nd F1max={b2[3].max():.2f} / classifier 2nd F1max={k2[3].max():.2f}",file=sys.stderr)

if __name__=="__main__": main()
