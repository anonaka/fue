#!/usr/bin/env python3
"""合成学習データ生成: ラベル付き試合から「明瞭な笛テンプレート」と「アンビエンス」を抜き、
笛を SNR・ピッチ・残響・定位 を変えてアンビエンスに混ぜ、弱い/遠い笛の難例を量産する。
v3 と同じ特徴を抽出して分類器の学習セット(npz: X, y)を保存する。

狙い: 実物の少ラベル(数十例)では学習できない「弱い笛 vs 弱いトーン性ノイズ」の分離を、
SNRや距離(残響)を制御した合成例で教える。プロトタイプで実物の赤の recall 56→81% を確認済み。

使い方: python3 synthesize.py <src.(m4a|wav)> <ground_truth.tsv> <out.npz> [n_per_pair=30]
  src はステレオ音源(中央=笛/横=観客 の定位を使う)。出力 npz は X(特徴), y(0/1), feat_names。
"""
import sys, os, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_v3 as D3, evaluate as E, cv_amateur as CV

SR, FPS = D3.SR, D3.FPS
P = dict(band_lo=1150., band_hi=2700., bg_sec=8., sustain_sec=0.10, pct=98.5, gap=0.6, side_penalty=2.0)
BAND = (1150., 2700.)
rng = np.random.default_rng(0)

def load_stereo(src, dur=None):
    cmd = ["ffmpeg","-v","error"] + (["-t",str(dur)] if dur else []) + ["-i",src,"-ac","2","-ar",str(SR),"-f","s16le","-"]
    a = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, "<i2").astype(np.float32).reshape(-1,2)/32768.
    return a[:,0].copy(), a[:,1].copy()

def band_rms(x, lo=BAND[0], hi=BAND[1]):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1/SR); X[(f<lo)|(f>hi)] = 0
    return float(np.sqrt((np.fft.irfft(X, n=len(x))**2).mean()) + 1e-9)

def pitch_shift(x, semitones):
    f = 2**(semitones/12.0); n = max(8, int(len(x)/f))
    return np.interp(np.linspace(0, len(x)-1, n), np.arange(len(x)), x).astype(np.float32)

def reverb(x, tau=0.05):
    """指数減衰の短いインパルス応答で残響=距離感を付加(L/Rを少し脱相関させる効果も)。"""
    n = int(0.12*SR); t = np.arange(n)/SR
    ir = (np.exp(-t/tau) * rng.standard_normal(n)).astype(np.float32); ir[0] = 1.0
    return np.convolve(x, ir)[:len(x)+n].astype(np.float32)

def stereo_candidates(L, R):
    mid = (L+R)/2; side = (L-R)/2
    mag, freqs = D3.spectrogram(mid); smag, _ = D3.spectrogram(side); T = mag.shape[0]
    wsel = (freqs>=P["band_lo"]) & (freqs<=P["band_hi"]); n = min(T, smag.shape[0])
    sm = np.zeros(T); sm[:n] = smag[:n,wsel].sum(1)/(mag[:n,wsel].sum(1)+1e-9)
    cands = D3.candidates(mag=mag, freqs=freqs, side_mag=smag)
    return mag, freqs, T, sm, cands

def feats(cands, mag, freqs, T, sm):
    X = CV.features(cands, P, mag, freqs, T, FPS)
    cen = np.array([sm[min(max(int(round(c[0]*FPS)),0),T-1)] for c in cands]).reshape(-1,1)
    return np.hstack([X, np.log1p(cen)])

def templates(mono, gt, n=8):
    mag, freqs = D3.spectrogram(mono)
    ridge, _ = D3.ridge_map(mag, freqs, P)
    s = ridge.max(1)
    score = lambda g: s[min(max(int(round(g*FPS)),0), len(s)-1)]
    HW = int(0.25*SR)
    out = []
    for g in sorted(gt, key=lambda g:-score(g))[:n]:
        seg = mono[max(0,int(g*SR)-HW):int(g*SR)+HW].copy()
        if len(seg) >= HW: out.append(seg*np.hanning(len(seg)).astype(np.float32))
    return out

def ambiences(L, R, gt, n=12, sec=14):
    out = []
    for _ in range(n*4):
        c = rng.uniform(sec, len(L)/SR - sec)
        if all(abs(c-g) > 9 for g in gt):
            a = int((c-sec/2)*SR); out.append((L[a:a+sec*SR].copy(), R[a:a+sec*SR].copy()))
        if len(out) >= n: break
    return out

SNRS = [-12,-9,-6,-3,0,6,12]

def synthesize(src, gtp, n_per_pair=30):
    L, R = load_stereo(src); mono = (L+R)/2; gt = E.load_gt(gtp)
    tmpls = templates(mono, gt); ambs = ambiences(L, R, gt)
    print(f"テンプレート{len(tmpls)} / アンビエンス{len(ambs)}", file=sys.stderr)
    X, y = [], []
    for k, (bL, bR) in enumerate(ambs):
        if len(bL) < 14*SR: continue
        arms = band_rms(bL[6*SR:8*SR])
        for _ in range(n_per_pair):
            t = tmpls[rng.integers(len(tmpls))]
            t = pitch_shift(t, rng.uniform(-3, 3))             # f0 多様化
            if rng.random() < 0.5: t = reverb(t)               # 距離(残響)
            t = (t * np.hanning(len(t)).astype(np.float32))
            snr = SNRS[rng.integers(len(SNRS))]
            pan = float(np.clip(rng.normal(0, 0.12), -0.35, 0.35))  # 主に中央、たまに少しずれ
            gain = 10**(snr/20.0) * arms / band_rms(t)
            ang = (pan+1)*np.pi/4; lg, rg = np.cos(ang)*gain*1.414, np.sin(ang)*gain*1.414
            L2, R2 = bL.copy(), bR.copy(); ins = 7*SR - len(t)//2
            L2[ins:ins+len(t)] += lg*t; R2[ins:ins+len(t)] += rg*t
            mag, freqs, T, sm, cands = stereo_candidates(L2, R2)
            if not cands: continue
            Xc = feats(cands, mag, freqs, T, sm); ct = [c[0] for c in cands]
            pi = int(np.argmin([abs(c-7.0) for c in ct]))
            if abs(ct[pi]-7.0) <= 0.6: X.append(Xc[pi]); y.append(1.)        # 挿入笛=正例
            for i in rng.permutation([j for j in range(len(cands)) if abs(ct[j]-7.0) > 1.5])[:3]:
                X.append(Xc[i]); y.append(0.)                                # アンビエンスのノイズ=負例
        print(f"  amb {k+1}/{len(ambs)}  累計 正{int(np.sum(y))}/負{int(len(y)-np.sum(y))}", file=sys.stderr)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("使い方: python3 synthesize.py <src.(m4a|wav)> <ground_truth.tsv> <out.npz> [n_per_pair=30]")
    npp = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    X, y = synthesize(sys.argv[1], sys.argv[2], npp)
    np.savez(sys.argv[3], X=X, y=y, feat_names=np.array(CV.FEAT + ["centering"]))
    print(f"保存: {sys.argv[3]}  X{X.shape} 正{int(y.sum())}/負{int((1-y).sum())}")
