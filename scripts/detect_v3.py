#!/usr/bin/env python3
"""笛検出 v3 — 横リッジ方式 (アマチュア/家庭用ビデオ録画向けにスクラッチ開発)。

放送データ(v1/v2)での知見:
  - 笛の判別情報は「per-frame の帯域エネルギーの大小」ではなく、スペクトログラム上の
    **時間方向に連続する細い横線(ridge)** にある(per-frame で帯域を1スカラーに潰すと失われる)。
  - 群衆PA/実況のある放送では ridge 比は中程度(例 1:57 で約7倍)。
アマチュア(家庭用ビデオ)データの実測(rakuwaku-yellow, 笛43):
  - 帯域突出(per-frame)は弱い(中央値3.7, 背景p99を超える笛は35%) → エネルギー検出は不利。
  - **ridge 比は桁違いに大きい(中央値57, 最大271)** → 笛の周波数行が背景に対し極めてクリーン。
  - f0 が二峰性: 高域~2400-2600Hz(多数) + 低域~1250-1620Hz(数件)。放送帯域1700-2500では低域を取り逃す。
→ v3 は ridge を**正式な候補生成器**として実装する(放送版が「次の一手」としていた方向)。

方式: 各周波数行をその行自身の時間背景(ロバスト)で正規化 → 時間方向に平滑(持続性) →
       笛帯域での行方向最大 = ridge スコア。ピークを候補に。per-frame エネルギーに依存しない。
"""
import subprocess, sys, os, numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR, NFFT, HOP = 24000, 1024, 512
FPS = SR / HOP

# 既定パラメータ(cross-validation で最適化する。下記は探索の初期値)
DEFAULTS = dict(
    # 既定はアマチュア試合(rakuwaku 黄/赤)の CV で検証した設定。
    band_lo=1150.0,    # 笛帯域下限(低f0の笛~1250Hzを拾う)
    band_hi=2700.0,    # 上限(高f0~2600 をカバー。3300は雑音増で不採用)
    bg_sec=8.0,        # 行背景の移動中央値の窓(笛の持続では動かない長さ)
    sustain_sec=0.10,  # 時間平滑(細く持続する横線を要求, 単発スパイクを抑制)
    pct=98.5,          # イベント抽出のスコア閾値パーセンタイル
    gap=0.6,           # 近接ピーク統合(秒)
    side_penalty=2.0,  # ステレオ centering: 横(side)に片寄るフレーム(横の掛け声)を減点する強さ
)

PROGRESS = None

def load(path):
    raw = subprocess.run(["ffmpeg","-v","error","-i",path,"-ac","1","-ar",str(SR),
                          "-f","s16le","-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32)/32768.0

def load_stereo(path):
    """ステレオで読み mid=(L+R)/2(=mono と同じ), side=(L-R)/2 を返す。
    mono/疑似ステレオ音源なら side≈0 になり centering は無影響。"""
    raw = subprocess.run(["ffmpeg","-v","error","-i",path,"-ac","2","-ar",str(SR),
                          "-f","s16le","-"], capture_output=True).stdout
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, 2)/32768.0
    return (a[:,0]+a[:,1])/2, (a[:,0]-a[:,1])/2

def spectrogram(x):
    win = np.hanning(NFFT).astype(np.float32); nf = 1+(len(x)-NFFT)//HOP
    freqs = np.fft.rfftfreq(NFFT, 1/SR); mag = np.empty((nf,len(freqs)), np.float32)
    for i in range(nf):
        mag[i] = np.abs(np.fft.rfft(x[i*HOP:i*HOP+NFFT]*win))
        if PROGRESS is not None and (i & 0x3FF)==0: PROGRESS(i,nf)
    if PROGRESS is not None: PROGRESS(nf,nf)
    return mag, freqs

def _movavg(v, win):
    c = np.concatenate([[0], np.cumsum(v)]); n = len(v); half = win//2
    lo = np.clip(np.arange(n)-half, 0, n); hi = np.clip(np.arange(n)+half+1, 0, n)
    return (c[hi]-c[lo])/(hi-lo)

def _row_background(M, bg_sec):
    """各行(周波数ビン)の時間背景をロバストに推定 (T,R)。
    ブロック中央値 → ブロック方向の移動中央値。短い笛は窓内で少数なので背景を押し上げない。"""
    T, R = M.shape
    BS = max(1, int(0.17*FPS))            # ~0.17s ブロック
    nb = T//BS
    blk = M[:nb*BS].reshape(nb, BS, R)
    bmed = np.median(blk, axis=1)         # (nb,R) ブロック内中央値(単発スパイク除去)
    WB = max(3, int(bg_sec*FPS/BS))       # 背景窓(ブロック数)
    pad = WB//2
    padded = np.pad(bmed, ((pad, WB-1-pad),(0,0)), mode="edge")
    win = sliding_window_view(padded, WB, axis=0)   # (nb,R,WB)
    bg_blk = np.median(win, axis=2)                  # (nb,R) 移動中央値
    bg = np.repeat(bg_blk, BS, axis=0)               # フレームへ展開
    if bg.shape[0] < T:                              # 端の余り
        bg = np.concatenate([bg, np.repeat(bg[-1:], T-bg.shape[0], axis=0)], axis=0)
    return bg[:T]

def ridge_map(mag, freqs, p):
    """笛帯域の ridge マップ(各行を時間背景で正規化, 持続平滑後) (T,R) と band_f を返す。"""
    sel = (freqs>=p["band_lo"]) & (freqs<=p["band_hi"])
    band_f = freqs[sel]
    M = mag[:, sel]                                  # (T,R)
    bg = _row_background(M, p["bg_sec"]) + 1e-6
    ridge = M/bg                                     # (T,R) 行ごと時間正規化
    k = max(1, int(p["sustain_sec"]*FPS))            # 持続平滑(時間方向移動平均)
    if k > 1:
        c = np.concatenate([np.zeros((1,ridge.shape[1])), np.cumsum(ridge,axis=0)])
        sm = (c[k:]-c[:-k])/k                         # (T-k+1,R)
        ridge = np.pad(sm, ((0, M.shape[0]-sm.shape[0]),(0,0)), mode="edge")
    return ridge, band_f

def ridge_score(mag, freqs, p):
    """ridge マップの行方向最大に「声(低域)ペナルティ」を掛けた per-frame スコアと、ピーク周波数。

    ridge は音量(マイクからの近さ)に比例するため、近くの観客の掛け声(大音量・声)が遠い主審の
    笛より高スコアになり誤検出を生む。掛け声は声の基本波(150-1200Hz)を多く含むので、その帯域の
    相対エネルギーでスコアを減点する(loudness非依存)。アマチュア2試合で F1 +5〜8pt を確認。
    """
    ridge, band_f = ridge_map(mag, freqs, p)
    score = ridge.max(axis=1)
    wsel = (freqs >= p["band_lo"]) & (freqs <= p["band_hi"])
    lowsel = (freqs >= 150) & (freqs <= 1200)
    low_ratio = mag[:, lowsel].sum(1) / (mag[:, wsel].sum(1) + 1e-9)
    score = score / (1.0 + low_ratio)
    return score, band_f[ridge.argmax(axis=1)]

def _centering(mag, side_mag, freqs, p):
    """ステレオ centering の重み (T,)。横(side)成分が多いフレーム(=横の観客の掛け声)を減点する。
    主審の笛はカメラ正面=中央(mid)で side≈小→温存、観客は横で side大→減点。mono なら side≈0→重み≈1。
    アマチュア実測: 笛の mid/side≈10 / 掛け声≈3.6。クロス試合で recall上限+ , F1 +5〜7pt を確認。"""
    wsel = (freqs >= p["band_lo"]) & (freqs <= p["band_hi"])
    n = min(mag.shape[0], side_mag.shape[0])
    w = np.ones(mag.shape[0])
    mid_b = mag[:n, wsel].sum(1) + 1e-9
    side_b = side_mag[:n, wsel].sum(1)
    w[:n] = 1.0 / (1.0 + p["side_penalty"] * (side_b / mid_b))
    return w

def _events(score, t, pct, gap):
    thr = np.percentile(score, pct); hot = np.where(score>thr)[0]; ev=[]
    if len(hot):
        s=q=hot[0]
        for i in hot[1:]:
            if (i-q)/FPS>gap: ev.append((s,q)); s=i
            q=i
        ev.append((s,q))
    out=[]
    for a,b in ev:
        k=a+score[a:b+1].argmax(); out.append((t[k], score[k]))
    return out

def candidates(path=None, params=None, mag=None, freqs=None, side_mag=None):
    """ridge スコア(+ステレオ centering)のピークを候補として返す: [(time, f0, score), ...] 高score順。
    path から読む場合はステレオで読み、横に片寄る掛け声を減点する(side_penalty)。
    mag/freqs を渡すと再計算を省く(CVで使い回す); その際 side_mag も渡せば centering を効かせる。"""
    p = dict(DEFAULTS)
    if params: p.update(params)
    if mag is None:
        mid, side = load_stereo(path)
        mag, freqs = spectrogram(mid)
        global PROGRESS
        _saved, PROGRESS = PROGRESS, None   # side のスペクトログラムは進捗に出さない
        try:
            side_mag, _ = spectrogram(side)
        finally:
            PROGRESS = _saved
    t=np.arange(mag.shape[0])/FPS
    score, peakf = ridge_score(mag, freqs, p)
    if side_mag is not None and p.get("side_penalty", 0) > 0:
        score = score * _centering(mag, side_mag, freqs, p)
    ev=_events(score, t, p["pct"], p["gap"])
    out=[]
    for tt, sc in ev:
        fr=int(round(tt*FPS)); fr=min(max(fr,0),len(peakf)-1)
        out.append((tt, float(peakf[fr]), float(sc)))
    out.sort(key=lambda r:-r[2])
    return out

# ---- 音色分類器 (production): 候補を whistle probability で再ランクする ----
FEAT = ["log_ridge","onset","dur","tonal","low_ratio","log_harm","f0","centering"]

def feature_vectors(cands, mag, freqs, side_mag=None, params=None):
    """候補ごとの音色特徴 (N,8): log_ridge/onset/dur/tonal/low_ratio/log_harm/f0/centering。
    side_mag を渡すと centering(横/中央) を入れる(mono は 0)。"""
    p = dict(DEFAULTS)
    if params: p.update(params)
    R, band_f = ridge_map(mag, freqs, p)
    s = R.max(1); T = mag.shape[0]; do = int(0.12*FPS)
    wsel = (freqs>=p["band_lo"]) & (freqs<=p["band_hi"]); lowsel = (freqs>=150) & (freqs<=1200)
    bandsum = mag[:,wsel].sum(1)+1e-9; bandpeak = mag[:,wsel].max(1); lowsum = mag[:,lowsel].sum(1)
    sm = np.zeros(T)
    if side_mag is not None:
        n = min(T, side_mag.shape[0]); sm[:n] = side_mag[:n,wsel].sum(1)/bandsum[:n]
    X = []
    for tt, f0, sc in cands:
        fr = min(max(int(round(tt*FPS)), 0), T-1)
        on = s[fr] - (s[fr-do] if fr-do >= 0 else s[fr])
        thr = 0.4*s[fr]; i = fr; nn = 0
        while i < T and s[i] > thr: nn += 1; i += 1
        i = fr-1
        while i >= 0 and s[i] > thr: nn += 1; i -= 1
        dur = nn/FPS
        ton = bandpeak[fr]/bandsum[fr]; low = lowsum[fr]/bandsum[fr]
        hf = 2*f0
        if p["band_lo"] <= hf <= p["band_hi"]:
            harm = R[fr, int(np.argmin(np.abs(band_f-hf)))]
        else:
            harm = mag[fr, int(np.argmin(np.abs(freqs-hf)))]/(mag[fr, int(np.argmin(np.abs(freqs-f0)))]+1e-9)
        X.append([np.log(sc+1), on, dur, ton, low, np.log(harm+1), f0/2500.0, np.log1p(sm[fr])])
    return np.array(X)

def train_classifier(X, y, iters=5000, lr=0.3, l2=1.0):
    mu = X.mean(0); sd = X.std(0)+1e-9; Xs = np.c_[np.ones(len(X)), (X-mu)/sd]; w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        pr = 1/(1+np.exp(-Xs@w)); g = Xs.T@(pr-y)/len(y); g[1:] += l2*w[1:]/len(y); w -= lr*g
    return dict(w=w, mu=mu, sd=sd)

def classify(X, model):
    Xs = np.c_[np.ones(len(X)), (X-model["mu"])/model["sd"]]
    return 1/(1+np.exp(-Xs@model["w"]))

def save_model(model, path): np.savez(path, w=model["w"], mu=model["mu"], sd=model["sd"])
def load_model(path):
    d = np.load(path); return dict(w=d["w"], mu=d["mu"], sd=d["sd"])

def detect_scored(path, model=None, params=None):
    """ステレオで候補生成し、model があれば whistle probability を value に、無ければ ridge スコアを返す。
    [(time, f0, value), ...] を高value順で返す。"""
    p = dict(DEFAULTS)
    if params: p.update(params)
    mid, side = load_stereo(path)
    mag, freqs = spectrogram(mid)
    global PROGRESS
    _saved, PROGRESS = PROGRESS, None         # side のスペクトログラムは進捗に出さない
    try: side_mag, _ = spectrogram(side)
    finally: PROGRESS = _saved
    cands = candidates(mag=mag, freqs=freqs, side_mag=side_mag, params=p)
    if model is None: return cands
    pr = classify(feature_vectors(cands, mag, freqs, side_mag, p), model)
    out = [(cands[i][0], cands[i][1], float(pr[i])) for i in range(len(cands))]
    out.sort(key=lambda r: -r[2])
    return out

if __name__=="__main__":
    if len(sys.argv)<2: sys.exit("使い方: python3 detect_v3.py <input.wav> [N]")
    n=int(sys.argv[2]) if len(sys.argv)>2 else 40
    cs=candidates(sys.argv[1])[:n]
    print("# mm:ss\tseconds\tf0(Hz)\tscore")
    for tt,f0,sc in sorted(cs, key=lambda r:r[0]):
        print(f"{int(tt//60)}:{tt%60:05.2f}\t{tt:.2f}\t{f0:.0f}\t{sc:.1f}")
