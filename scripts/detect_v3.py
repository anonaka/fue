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
    band_lo=1150.0,    # 笛帯域下限(低f0の笛~1250Hzを拾う)
    band_hi=3300.0,    # 上限(高f0~2600 + 一部~3100をカバー)
    bg_sec=6.0,        # 行背景の移動中央値の窓(笛の持続では動かない長さ)
    sustain_sec=0.08,  # 時間平滑(細く持続する横線を要求, 単発スパイクを抑制)
    pct=99.0,          # イベント抽出のスコア閾値パーセンタイル
    gap=0.6,           # 近接ピーク統合(秒)
)

PROGRESS = None

def load(path):
    raw = subprocess.run(["ffmpeg","-v","error","-i",path,"-ac","1","-ar",str(SR),
                          "-f","s16le","-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32)/32768.0

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

def candidates(path=None, params=None, mag=None, freqs=None):
    """ridge スコアのピークを候補として返す: [(time, f0, score), ...] 高score順。
    mag/freqs を渡すと再計算を省く(CVで使い回す)。"""
    p = dict(DEFAULTS);
    if params: p.update(params)
    if mag is None:
        x=load(path); mag,freqs=spectrogram(x)
    t=np.arange(mag.shape[0])/FPS
    score, peakf = ridge_score(mag, freqs, p)
    ev=_events(score, t, p["pct"], p["gap"])
    out=[]
    for tt, sc in ev:
        fr=int(round(tt*FPS)); fr=min(max(fr,0),len(peakf)-1)
        out.append((tt, float(peakf[fr]), float(sc)))
    out.sort(key=lambda r:-r[2])
    return out

if __name__=="__main__":
    if len(sys.argv)<2: sys.exit("使い方: python3 detect_v3.py <input.wav> [N]")
    n=int(sys.argv[2]) if len(sys.argv)>2 else 40
    cs=candidates(sys.argv[1])[:n]
    print("# mm:ss\tseconds\tf0(Hz)\tscore")
    for tt,f0,sc in sorted(cs, key=lambda r:r[0]):
        print(f"{int(tt//60)}:{tt%60:05.2f}\t{tt:.2f}\t{f0:.0f}\t{sc:.1f}")
