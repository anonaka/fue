#!/usr/bin/env python3
"""音色分類器を学習して npz(w, mu, sd)に保存する。fue.detect が読み込み候補を whistle probability で再ランクする。

学習データ = 実試合(ステレオ音源+正解tsv) と/または 合成データ(synthesize.py の npz)。
ステレオ音源で学習すること(centering 特徴が要るため)。in-domain の試合ほど効く。

使い方:
  python3 train_classifier.py <out.npz> <src.(m4a|mov|wav)>[:dur] <gt.tsv> [<src>[:dur] <gt> ...] [--synth a.npz ...]
  例: train_classifier.py ../models/whistle_clf.npz yellow.m4a yellow_gt.tsv red.m4a:600 red_gt.tsv --synth synth.npz
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_v3 as D3, evaluate as E

def from_match(src, gt, dur=None):
    raw_src = src
    mid, side = (D3.load_stereo(src) if dur is None else _load_stereo_dur(src, dur))
    mag, freqs = D3.spectrogram(mid)
    smag, _ = D3.spectrogram(side)
    cands = D3.candidates(mag=mag, freqs=freqs, side_mag=smag)
    X = D3.feature_vectors(cands, mag, freqs, smag)
    gts = E.load_gt(gt)
    y = np.array([1.0 if any(abs(c[0]-g) <= E.TOL for g in gts) else 0.0 for c in cands])
    print(f"  {os.path.basename(raw_src)}: 候補{len(cands)} 正例{int(y.sum())}", file=sys.stderr)
    return X, y

def _load_stereo_dur(src, dur):
    import subprocess
    raw = subprocess.run(["ffmpeg","-v","error","-t",str(dur),"-i",src,"-ac","2","-ar",str(D3.SR),"-f","s16le","-"],
                         capture_output=True).stdout
    a = np.frombuffer(raw, "<i2").astype(np.float32).reshape(-1,2)/32768.0
    return (a[:,0]+a[:,1])/2, (a[:,0]-a[:,1])/2

def main(argv):
    out = argv[0]; rest = argv[1:]
    synth = []
    if "--synth" in rest:
        k = rest.index("--synth"); synth = rest[k+1:]; rest = rest[:k]
    if len(rest) % 2 != 0:
        sys.exit("使い方: train_classifier.py <out.npz> <src>[:dur] <gt> [...] [--synth a.npz ...]")
    Xs, ys = [], []
    for i in range(0, len(rest), 2):
        spec, gt = rest[i], rest[i+1]
        src, dur = (spec.rsplit(":",1)[0], float(spec.rsplit(":",1)[1])) if ":" in spec and spec.rsplit(":",1)[1].replace(".","").isdigit() else (spec, None)
        X, y = from_match(src, gt, dur); Xs.append(X); ys.append(y)
    for npz in synth:
        d = np.load(npz); Xs.append(d["X"]); ys.append(d["y"])
        print(f"  synth {os.path.basename(npz)}: {len(d['y'])}例 正{int(d['y'].sum())}", file=sys.stderr)
    X = np.vstack(Xs); y = np.concatenate(ys)
    model = D3.train_classifier(X, y)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    D3.save_model(model, out)
    print(f"保存: {out}  学習 {len(y)}例(正{int(y.sum())}/負{int((1-y).sum())}) 特徴{X.shape[1]}次元 {D3.FEAT}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("使い方: python3 train_classifier.py <out.npz> <src>[:dur] <gt.tsv> [...] [--synth a.npz ...]")
    main(sys.argv[1:])
