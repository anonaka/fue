#!/usr/bin/env python3
"""アマチュア試合を前半/後半に分け、v3(横リッジ)検出器を 2-fold クロスバリデーションで評価する。

目的: 家庭用ビデオ録画(アマチュア)の笛検出を、放送データの知見を踏まえスクラッチ開発した
v3 で評価。1試合を時間で前半/後半に分割し、(A)前半学習→後半評価 (B)後半学習→前半評価。
各 fold で
  1) 検出器パラメータを train 側のみで最適化(グリッド探索, F1最良で選択),
  2) 音色分類器(ロジスティック回帰)を train 候補で学習し test 候補を再ランク,
honest に test 側で評価する。

使い方: python3 cv_amateur.py <wav> <ground_truth.tsv>
"""
import sys, os, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_v3 as D3, evaluate as E

TOL = E.TOL  # 4.0s

# ---- 検出器パラメータのグリッド ----
GRID = dict(band_lo=[1100.0, 1300.0, 1600.0], band_hi=[2700.0, 3300.0],
            bg_sec=[5.0, 8.0], sustain_sec=[0.06, 0.10], pct=[98.5, 99.0])

def f1_curve(times, gtsub):
    """test/train サブセットに対する (F1最良, N, P, R, 上限recall)。"""
    if not times or not gtsub: return 0.0, 0, 0.0, 0.0, 0.0
    _, used = E.match(times, gtsub); ceil = sum(used)/len(gtsub)
    best = (0.0, 0, 0.0, 0.0)
    for N, tp, fp, fn, P, R, F in E.curve(times, gtsub):
        if F > best[0]: best = (F, N, P, R)
    return best[0], best[1], best[2], best[3], ceil

# ---- 音色分類器 (numpy ロジスティック回帰) ----
def logreg_train(X, y, iters=5000, lr=0.3, l2=1.0):
    mu = X.mean(0); sd = X.std(0)+1e-9; Xs = np.c_[np.ones(len(X)), (X-mu)/sd]
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = 1/(1+np.exp(-Xs@w)); g = Xs.T@(p-y)/len(y); g[1:] += l2*w[1:]/len(y); w -= lr*g
    return w, mu, sd

def logreg_pred(X, w, mu, sd):
    return 1/(1+np.exp(-np.c_[np.ones(len(X)), (X-mu)/sd]@w))

FEAT = ["log_ridge","onset","dur","tonal","low_ratio","log_harm","f0"]

def features(cands, params, mag, freqs, T, FPS):
    R, band_f = D3.ridge_map(mag, freqs, params)
    s = R.max(1); do = int(0.12*FPS)
    wsel = (freqs>=params["band_lo"]) & (freqs<=params["band_hi"])
    lowsel = (freqs>=150) & (freqs<=1200)
    bandsum = mag[:,wsel].sum(1)+1e-9; bandpeak = mag[:,wsel].max(1)
    lowsum = mag[:,lowsel].sum(1)
    X = []
    for tt, f0, sc in cands:
        fr = min(max(int(round(tt*FPS)), 0), T-1)
        on = s[fr] - (s[fr-do] if fr-do >= 0 else s[fr])
        thr = 0.4*s[fr]; i = fr; n = 0
        while i < T and s[i] > thr: n += 1; i += 1
        i = fr-1
        while i >= 0 and s[i] > thr: n += 1; i -= 1
        dur = n/FPS
        ton = bandpeak[fr]/bandsum[fr]; low = lowsum[fr]/bandsum[fr]
        hf = 2*f0
        if params["band_lo"] <= hf <= params["band_hi"]:
            harm = R[fr, int(np.argmin(np.abs(band_f-hf)))]
        else:
            harm = mag[fr, int(np.argmin(np.abs(freqs-hf)))]/(mag[fr, int(np.argmin(np.abs(freqs-f0)))]+1e-9)
        X.append([np.log(sc+1), on, dur, ton, low, np.log(harm+1), f0/2500.0])
    return np.array(X)

def label(cands, gtsub):
    y = []
    for tt, _, _ in cands:
        y.append(1.0 if any(abs(tt-g) <= TOL for g in gtsub) else 0.0)
    return np.array(y)

def main(wav, gtpath):
    x = D3.load(wav); mag, freqs = D3.spectrogram(x); FPS = D3.FPS; T = mag.shape[0]
    dur = T/FPS; MID = dur/2
    gt = E.load_gt(gtpath)
    gt1 = [g for g in gt if g < MID]; gt2 = [g for g in gt if g >= MID]
    print(f"試合長 {dur:.0f}s / 笛 {len(gt)}個 (前半 {len(gt1)} / 後半 {len(gt2)}), 分割点 {MID:.0f}s\n")

    # 全パラメータ組の候補を一度だけ生成してキャッシュ(fold 間で共有)
    keys = list(GRID); combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    print(f"パラメータ探索 {len(combos)} 通りの候補を生成中...", file=sys.stderr)
    cache = {}
    for i, p in enumerate(combos):
        key = tuple(p[k] for k in keys)
        cache[key] = D3.candidates(mag=mag, freqs=freqs, params=p)
        if (i+1) % 8 == 0: print(f"  {i+1}/{len(combos)}", file=sys.stderr)

    def split(cands):
        return [c for c in cands if c[0] < MID], [c for c in cands if c[0] >= MID]

    folds = [("A: 前半学習→後半評価", gt1, gt2, "test2"),
             ("B: 後半学習→前半評価", gt2, gt1, "test1")]
    agg = {"ceil":[], "det":[], "clf":[]}
    for name, gtr, gte, which in folds:
        # 1) train 側のみでパラメータ最適化
        best = None
        for key in cache:
            c1, c2 = split(cache[key])
            tr = c2 if which == "test1" else c1     # train 領域の候補
            ttr = [c[0] for c in tr]
            f1, N, P, Rr, ceil = f1_curve(ttr, gtr)
            if best is None or f1 > best[0]: best = (f1, key, N)
        bf1, bkey, bN = best; bp = dict(zip(keys, bkey))
        cands = cache[bkey]; c1, c2 = split(cands)
        tr = c2 if which == "test1" else c1
        te = c1 if which == "test1" else c2
        # 2) 検出器のみ(ridge スコア順)で test 評価
        tte = [c[0] for c in te]
        dF, dN, dP, dR, ceil = f1_curve(tte, gte)
        # 3) 分類器: train 候補で学習 → test 候補を再ランク
        Xall = features(cands, bp, mag, freqs, T, FPS)
        idx = {id(c): i for i, c in enumerate(cands)}
        itr = [idx[id(c)] for c in tr]; ite = [idx[id(c)] for c in te]
        ytr = label(tr, gtr)
        w, mu, sd = logreg_train(Xall[itr], ytr)
        pte = logreg_pred(Xall[ite], w, mu, sd)
        order = [te[i][0] for i in np.argsort(-pte)]
        cF, cN, cP, cR, _ = f1_curve(order, gte)
        print(f"[{name}]")
        print(f"  選択パラメータ: band={bp['band_lo']:.0f}-{bp['band_hi']:.0f}Hz "
              f"bg={bp['bg_sec']:.0f}s sustain={bp['sustain_sec']:.2f}s pct={bp['pct']} (train F1={bf1*100:.0f}%)")
        print(f"  test候補 {len(te)} / 笛 {len(gte)} / 上限recall {ceil*100:.0f}%")
        print(f"  検出器のみ(ridge順) : F1 {dF*100:.0f}% (N={dN}, P{dP*100:.0f}/R{dR*100:.0f})")
        print(f"  +音色分類器(再ランク): F1 {cF*100:.0f}% (N={cN}, P{cP*100:.0f}/R{cR*100:.0f})\n")
        agg["ceil"].append(ceil); agg["det"].append(dF); agg["clf"].append(cF)

    print("=== 2-fold 平均 (test 側) ===")
    print(f"  上限recall      : {np.mean(agg['ceil'])*100:.0f}%")
    print(f"  検出器のみ F1   : {np.mean(agg['det'])*100:.0f}%")
    print(f"  +分類器 F1      : {np.mean(agg['clf'])*100:.0f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("使い方: python3 cv_amateur.py <wav> <ground_truth.tsv>")
    main(sys.argv[1], sys.argv[2])
