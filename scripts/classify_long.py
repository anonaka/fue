#!/usr/bin/env python3
"""「長い笛は拾う・短い笛は見逃して良い」運用での検出を評価する実験。

狙い(NOTES「実験: 横リッジ検出」「笛の長さ実測」より):
  - エネルギー候補が落とす長い笛は 1:57(0.47s) ただ1個。→ 横リッジ候補で和集合し回収。
  - 短い笛は見逃して良いので「最低持続長ゲート」で短い誤検出を切り precision を上げる。
  - 評価対象を「長い笛(>=DUR_LONG秒)」に再定義し、F1 を baseline と比較する。

評価指標(長笛運用):
  recall    = 拾えた長い笛 / 長い笛の総数   (短い笛は分母から除外=見逃して良い)
  precision = いずれかの笛に当たった候補 / 採用候補数 (短い笛に当たっても誤検出にしない)
"""
import os, sys, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_whistle_v2 as V2, detect_whistle as D, evaluate as E, classify as C

FPS=V2.FPS
TOL=E.TOL
DUR_LONG=0.40       # これ以上を「長い笛」とみなす(全47個中15個)
GATE=0.0            # 候補の最低持続長ゲート(0=無効)。短い誤検出を切るのに使う

# ---------- 笛/候補の持続長を測る(f0行が局所中央値の1.5倍を超える連続長) ----------
def sustain(mag,freqs,t,f0=None):
    c=int(t*FPS); band=(freqs>=1700)&(freqs<=2600)
    if f0 and f0>0: bi=int(np.argmin(np.abs(freqs-f0)))
    else:
        a,b=max(0,c-2),c+3; bi=int(np.where(band)[0][mag[a:b,band].mean(0).argmax()])
    row=mag[:,bi]; med=np.median(row[max(0,c-int(3*FPS)):c+int(3*FPS)])+1e-9; thr=1.5*med
    w=range(max(0,c-int(0.5*FPS)),min(len(row),c+int(0.5*FPS)))
    cc=max(w,key=lambda i:row[i]); lo=hi=cc
    while lo>0 and row[lo-1]>thr: lo-=1
    while hi<len(row)-1 and row[hi+1]>thr: hi+=1
    return (hi-lo+1)/FPS

# ---------- 倍音つき横リッジの候補生成器(プール外の長い笛を回収) ----------
def ridge_candidates(mag,freqs,topn=120):
    """各周波数行を自分の時間背景(4s)で正規化→時間0.5s平滑(持続)→フレーム内中央値を
    引く(周波数の細さ)→ f0線 + 0.5*2倍音線 を最大化。NMS で時刻候補を返す。"""
    def normband(lo,hi):
        m=(freqs>=lo)&(freqs<=hi); Bb=mag[:,m]; ff=freqs[m]; F=Bb.shape[1]
        k=np.ones(int(4.0*FPS))/int(4.0*FPS)
        bg=np.vstack([np.convolve(Bb[:,j],k,'same') for j in range(F)]).T+1e-9
        kt=np.ones(int(0.5*FPS))/int(0.5*FPS)
        Cc=Bb/bg; Cc=np.vstack([np.convolve(Cc[:,j],kt,'same') for j in range(F)]).T
        return np.clip(Cc-np.median(Cc,1,keepdims=True),0,None), ff
    Cf,bf=normband(1500,2800)        # 基本波帯
    Ch,hf=normband(3000,5200)        # 第2倍音帯
    h2=np.array([np.argmin(np.abs(hf-2*f0)) for f0 in bf])
    S=Cf+0.5*Ch[:,h2]                # フレーム×f0 の合議ライン強度
    score=S.max(1); bestf=bf[S.argmax(1)]
    order=np.argsort(score)[::-1]; out=[]; taken=[]
    for i in order:
        tt=i/FPS
        if any(abs(tt-a)<2.0 for a in taken): continue
        taken.append(tt); out.append((tt,float(bestf[i])))
        if len(out)>=topn: break
    return out

def main(path, gt_path):
    x=V2.load(path); mag,freqs=V2.spectrogram(x)
    gt=E.load_gt(gt_path)
    # 正解の長さ→長い笛集合
    gdur={g:sustain(mag,freqs,g) for g in gt}
    longs=[g for g in gt if gdur[g]>=DUR_LONG]
    print(f"正解 {len(gt)}個 / 長い笛(>= {DUR_LONG}s) {len(longs)}個",file=sys.stderr)

    # 候補プール: energy ∪ ridge
    ener=[(c[0],c[1]) for c in D.candidates(path)]
    rid =ridge_candidates(mag,freqs,topn=120)
    # 少数精鋭: 長いリッジ(持続>=DUR_LONG)だけに絞る → プールを汚さず長笛だけ足す
    rid_long=[(t,f) for t,f in rid if sustain(mag,freqs,t,f)>=DUR_LONG]
    print(f"ridge候補 {len(rid)} → 長いリッジのみ {len(rid_long)}",file=sys.stderr)
    def union(pool_extra):
        pool=list(ener)
        for t,f in pool_extra:
            if not any(abs(t-pt)<2.0 for pt,_ in pool): pool.append((t,f))
        return pool

    def recall_long(times):
        return sum(any(abs(g-t)<TOL for t in times) for g in longs)

    print("\n--- 長い笛の再現率上限(候補プール) ---")
    print(f"energy単独            : {recall_long([t for t,_ in ener])}/{len(longs)}")
    print(f"energy ∪ ridge(全120) : {recall_long([t for t,_ in union(rid)])}/{len(longs)}")
    print(f"energy ∪ 長ridgeのみ  : {recall_long([t for t,_ in union(rid_long)])}/{len(longs)}")

    # ---- 分類器評価(baseline=energyのみ・既存特徴 / new=union・持続長特徴+ゲート) ----
    def evaluate(pool, add_dur, gate):
        ts=[t for t,_ in pool]; fs=[f for _,f in pool]
        if gate>0:
            keep=[i for i,t in enumerate(ts) if sustain(mag,freqs,t,fs[i])>=gate]
            ts=[ts[i] for i in keep]; fs=[fs[i] for i in keep]
        X=np.array([C.feats(mag,freqs,t) for t in ts])
        if add_dur:
            d=np.array([[sustain(mag,freqs,t,f)] for t,f in zip(ts,fs)])
            X=np.c_[X,d]
        y=np.array([1.0 if any(abs(t-g)<=TOL for g in gt) else 0.0 for t in ts])  # 笛(長短問わず)
        oof=C.logreg_cv(X,y)
        order=np.argsort(-oof); rk=[(ts[i],fs[i],oof[i]) for i in order]
        # 長笛運用の P/R/F1 を N掃引
        best=(0,0,0,0)
        usedL=[False]*len(longs)
        rows=[]
        anyhit=0
        for N in range(1,len(rk)+1):
            t,_,_=rk[N-1]
            anyhit=sum(1 for tt,_,_ in rk[:N] if any(abs(tt-g)<TOL for g in gt))  # 何かに当たった候補数
            rl=sum(any(abs(g-tt)<TOL for tt,_,_ in rk[:N]) for g in longs)
            P=anyhit/N; R=rl/len(longs); F=2*P*R/(P+R) if P+R else 0
            rows.append((N,P,R,F))
            if F>best[3]: best=(N,P,R,F)
        return best,rk

    print("\n--- 長笛運用での分類器F1(交差検証) ---")
    (b0),_=evaluate(ener, add_dur=False, gate=0.0)
    print(f"baseline (energy, 既存11特徴)          : F1最良 {b0[3]*100:.0f}%  (N={b0[0]} P{b0[1]*100:.0f}/R{b0[2]*100:.0f})")
    (bd),_=evaluate(ener, add_dur=True, gate=0.0)
    print(f"energy + 持続長特徴のみ                 : F1最良 {bd[3]*100:.0f}%  (N={bd[0]} P{bd[1]*100:.0f}/R{bd[2]*100:.0f})")
    (b1),_=evaluate(union(rid), add_dur=True, gate=0.0)
    print(f"+ridge全120 +持続長特徴                 : F1最良 {b1[3]*100:.0f}%  (N={b1[0]} P{b1[1]*100:.0f}/R{b1[2]*100:.0f})")
    (b2),_=evaluate(union(rid_long), add_dur=True, gate=0.0)
    print(f"+長ridgeのみ +持続長特徴 (本命)         : F1最良 {b2[3]*100:.0f}%  (N={b2[0]} P{b2[1]*100:.0f}/R{b2[2]*100:.0f})")

if __name__=="__main__":
    if len(sys.argv)<3: sys.exit("使い方: python3 classify_long.py <input.wav> <ground_truth.tsv>")
    main(sys.argv[1], sys.argv[2])
