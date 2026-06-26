#!/usr/bin/env python3
"""レフェリーの笛をwavから検出する。

笛は約1.7-2.5kHzの基本波 + 約2倍の倍音を持つ、持続した狭帯域トーン。
観客ノイズ（広帯域）と区別するため、
  (1) 基本波帯域のトーン性(エネルギー集中度)
  (2) 倍音帯域との同時エネルギー
  (3) 低域(観客)に対するコントラスト
  (4) 一定時間の持続
を組み合わせた3種の指標で検出し、合議でランク付けする。

使い方:  python3 detect_whistle_v1.py <input.wav> [N]
         N = 出力する上位件数 (既定20)
"""
import subprocess, sys, os, numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SR, NFFT, HOP = 12000, 1024, 256
FPS = SR / HOP

# 進捗通知フック。呼び出し側が PROGRESS = func(done, total) を設定すると、
# spectrogram の計算中に進捗が通知される（既定 None なら何もしない）。
PROGRESS = None

def load(path):
    raw = subprocess.run(["ffmpeg","-v","error","-i",path,"-ac","1",
                          "-ar",str(SR),"-f","s16le","-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32)/32768.0

def spectrogram(x):
    win = np.hanning(NFFT).astype(np.float32)
    nf = 1 + (len(x)-NFFT)//HOP
    freqs = np.fft.rfftfreq(NFFT, 1/SR)
    mag = np.empty((nf, len(freqs)), np.float32)
    for i in range(nf):
        mag[i] = np.abs(np.fft.rfft(x[i*HOP:i*HOP+NFFT]*win))
        if PROGRESS is not None and (i & 0x3FF) == 0:
            PROGRESS(i, nf)
    if PROGRESS is not None:
        PROGRESS(nf, nf)
    return mag, freqs

def rollmin(a, sec):
    w = int(sec*FPS); pad = w//2
    return sliding_window_view(np.pad(a,(pad,w-1-pad),mode="edge"), w).min(1)

def events(score, t, pct=99.2, gap=1.0):
    thr = np.percentile(score, pct); hot = np.where(score>thr)[0]; ev=[]
    if len(hot):
        s=p=hot[0]
        for i in hot[1:]:
            if (i-p)/FPS>gap: ev.append((s,p)); s=i
            p=i
        ev.append((s,p))
    return [(t[a+score[a:b+1].argmax()], score[a+score[a:b+1].argmax()]) for a,b in ev]

def detect(path, topn=20):
    x = load(path); mag, freqs = spectrogram(x)
    t = np.arange(mag.shape[0])*HOP/SR
    bmax = lambda lo,hi: (mag[:,(freqs>=lo)&(freqs<=hi)].max(1),
                          freqs[(freqs>=lo)&(freqs<=hi)][mag[:,(freqs>=lo)&(freqs<=hi)].argmax(1)])
    floor = mag[:,(freqs>=150)&(freqs<=1300)].mean(1)+1e-9
    f0e,f0f = bmax(1700,2500); h2e,_ = bmax(3500,5000)
    tB = f0e/(mag[:,(freqs>=1700)&(freqs<=2500)].sum(1)+1e-9)
    MA = mag[:,(freqs>=1800)&(freqs<=5000)]; pA = MA.max(1)
    A = rollmin(pA*(pA/(MA.sum(1)+1e-9))*(pA/floor), 0.25)        # 帯域トーン
    B = rollmin(np.sqrt(f0e*h2e)/floor*tB, 0.25)                  # ハーモニクス
    C = rollmin(f0e/floor*tB, 0.30)                               # 基本波持続

    def ranked(ev, N=30):
        ev = sorted(ev, key=lambda r:-r[1])[:N]
        return [(tt, N-i) for i,(tt,_) in enumerate(ev)]
    cands = {}
    for ev in (ranked(events(A,t)), ranked(events(B,t)), ranked(events(C,t))):
        for tt,pts in ev:
            key = next((k for k in cands if abs(k-tt)<1.2), None)
            if key is None: cands[tt] = [pts,1,f0f[int(round(tt*FPS))]]
            else: cands[key][0]+=pts; cands[key][1]+=1
    final = sorted(([tt,*v] for tt,v in cands.items()), key=lambda r:(-r[2],-r[1]))
    return sorted(final[:topn], key=lambda r:r[0])

if __name__ == "__main__":
    if len(sys.argv)<2: sys.exit("使い方: python3 detect_whistle_v1.py <input.wav> [N]")
    path = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv)>2 else 20
    print("# mm:ss\tseconds\tf0(Hz)")
    for tt,pts,nd,ff in detect(path, topn):
        print(f"{int(tt//60)}:{tt%60:05.2f}\t{tt:.2f}\t{ff:.0f}")
