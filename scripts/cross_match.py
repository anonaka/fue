#!/usr/bin/env python3
"""2試合間のクロス評価: train試合の候補で音色分類器を学習し、test試合の候補を再ランクして評価。

別試合への汎化(別対戦カード・別主審/会場の可能性)を検証する。v3(横リッジ)で候補生成し、
cv_amateur の特徴量・ロジ回帰を流用。両方向(A→B, B→A)を評価する。

使い方: python3 cross_match.py <A.wav> <A_gt.tsv> <B.wav> <B_gt.tsv>
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_v3 as D3, evaluate as E, cv_amateur as CV

# アマチュア向けに CV で得た検出器パラメータ(band下限を放送1700より下げる)
PARAMS = dict(band_lo=1150.0, band_hi=2700.0, bg_sec=8.0, sustain_sec=0.10, pct=98.5, gap=0.6)

def prep(wav, gtpath):
    x = D3.load(wav); mag, freqs = D3.spectrogram(x); T = mag.shape[0]; FPS = D3.FPS
    gt = E.load_gt(gtpath)
    cands = D3.candidates(mag=mag, freqs=freqs, params=PARAMS)
    X = CV.features(cands, PARAMS, mag, freqs, T, FPS)
    y = CV.label(cands, gt)
    return dict(name=os.path.splitext(os.path.basename(wav))[0],
                cands=cands, X=X, y=y, gt=gt)

def cross(train, test):
    w, mu, sd = CV.logreg_train(train["X"], train["y"])
    p = CV.logreg_pred(test["X"], w, mu, sd)
    times = [c[0] for c in test["cands"]]
    dF, dN, dP, dR, ceil = CV.f1_curve(times, test["gt"])          # 検出器のみ(ridge順)
    order = [test["cands"][i][0] for i in np.argsort(-p)]
    cF, cN, cP, cR, _ = CV.f1_curve(order, test["gt"])             # 分類器で再ランク
    print(f"[{train['name']} で学習 → {test['name']} で評価]")
    print(f"  test候補 {len(times)} / 笛 {len(test['gt'])} / 上限recall {ceil*100:.0f}%")
    print(f"  検出器のみ(ridge順) : F1 {dF*100:.0f}% (N={dN}, P{dP*100:.0f}/R{dR*100:.0f})")
    print(f"  +分類器(別試合学習) : F1 {cF*100:.0f}% (N={cN}, P{cP*100:.0f}/R{cR*100:.0f})\n")

def main(aw, ag, bw, bg):
    A = prep(aw, ag); B = prep(bw, bg)
    print(f"A={A['name']} (笛{len(A['gt'])}, 候補{len(A['cands'])}) / "
          f"B={B['name']} (笛{len(B['gt'])}, 候補{len(B['cands'])})\n")
    cross(A, B)
    cross(B, A)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit("使い方: python3 cross_match.py <A.wav> <A_gt.tsv> <B.wav> <B_gt.tsv>")
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
