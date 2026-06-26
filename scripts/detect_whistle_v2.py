#!/usr/bin/env python3
"""笛検出 v2 (耳の周波数選択性を反映)。

v1は「笛 ÷ 低域(150-1300Hz)の群衆レベル」で対比を取るため、群衆が大きい場面の
弱い笛(例 1:57)を取り逃した。人の耳は低域の群衆では2kHzの笛をマスクしない
(臨界帯域=周波数選択性)。そこで分母を「笛帯域そのものの局所背景(時間方向)」に変更。
v1の3指標(帯域トーン/倍音/基本波) + 合議の枠組みは踏襲し、24kHzで解析、持続も短縮。
"""
import subprocess, sys, os, numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR, NFFT, HOP = 24000, 1024, 512
FPS = SR/HOP
SUSTAIN = 0.10   # 短い笛も拾えるよう短く

# 進捗通知フック（呼び出し側が PROGRESS = func(done, total) を設定）
PROGRESS = None

def load(path):
    raw=subprocess.run(["ffmpeg","-v","error","-i",path,"-ac","1","-ar",str(SR),
                        "-f","s16le","-"],capture_output=True).stdout
    return np.frombuffer(raw,dtype="<i2").astype(np.float32)/32768.0

def spectrogram(x):
    win=np.hanning(NFFT).astype(np.float32); nf=1+(len(x)-NFFT)//HOP
    freqs=np.fft.rfftfreq(NFFT,1/SR); mag=np.empty((nf,len(freqs)),np.float32)
    for i in range(nf):
        mag[i]=np.abs(np.fft.rfft(x[i*HOP:i*HOP+NFFT]*win))
        if PROGRESS is not None and (i & 0x3FF)==0: PROGRESS(i,nf)
    if PROGRESS is not None: PROGRESS(nf,nf)
    return mag,freqs

def movmean(v,win):
    c=np.concatenate([[0],np.cumsum(v)]); n=len(v); half=win//2
    lo=np.clip(np.arange(n)-half,0,n); hi=np.clip(np.arange(n)+half+1,0,n)
    return (c[hi]-c[lo])/(hi-lo)

def rollmin(a,sec):
    w=max(1,int(sec*FPS)); pad=w//2
    return sliding_window_view(np.pad(a,(pad,w-1-pad),mode="edge"),w).min(1)

def events(score,t,pct=99.2,gap=0.6):
    thr=np.percentile(score,pct); hot=np.where(score>thr)[0]; ev=[]
    if len(hot):
        s=p=hot[0]
        for i in hot[1:]:
            if (i-p)/FPS>gap: ev.append((s,p)); s=i
            p=i
        ev.append((s,p))
    out=[]
    for a,b in ev:
        k=a+score[a:b+1].argmax(); out.append((t[k],score[k]))
    return out

def detect(path,topn=25):
    x=load(path); mag,freqs=spectrogram(x); t=np.arange(mag.shape[0])/FPS
    bmax=lambda lo,hi:(mag[:,(freqs>=lo)&(freqs<=hi)].max(1),
                       freqs[(freqs>=lo)&(freqs<=hi)][mag[:,(freqs>=lo)&(freqs<=hi)].argmax(1)])
    f0e,f0f=bmax(1700,2500); h2e,_=bmax(3500,5000)
    tB=f0e/(mag[:,(freqs>=1700)&(freqs<=2500)].sum(1)+1e-9)
    MA=mag[:,(freqs>=1800)&(freqs<=5000)]; pA=MA.max(1)
    # ★ 分母 = 笛帯域そのものの局所背景(約4秒の時間移動平均)。群衆の低域には依存しない
    bg=movmean(f0e,int(4.0*FPS))+1e-9
    A=rollmin(pA*(pA/(MA.sum(1)+1e-9))*(pA/bg),SUSTAIN)
    B=rollmin(np.sqrt(f0e*h2e)/bg*tB,SUSTAIN)
    C=rollmin(f0e/bg*tB,SUSTAIN)
    def ranked(ev,N=60): ev=sorted(ev,key=lambda r:-r[1])[:N]; return [(tt,N-i) for i,(tt,_) in enumerate(ev)]
    cands={}
    for ev in (ranked(events(A,t)),ranked(events(B,t)),ranked(events(C,t))):
        for tt,pts in ev:
            key=next((k for k in cands if abs(k-tt)<1.0),None)
            if key is None: cands[tt]=[pts,1,f0f[int(round(tt*FPS))]]
            else: cands[key][0]+=pts; cands[key][1]+=1
    final=sorted(([tt,*v] for tt,v in cands.items()),key=lambda r:(-r[2],-r[1]))
    return [(tt,ff,pts,nd) for tt,pts,nd,ff in final[:topn]]

if __name__=="__main__":
    if len(sys.argv)<2: sys.exit("使い方: python3 detect_whistle_v2.py <input.wav> [N]")
    path=sys.argv[1]
    topn=int(sys.argv[2]) if len(sys.argv)>2 else 25
    print("# mm:ss\tseconds\tf0(Hz)")
    for tt,ff,pts,nd in sorted(detect(path,topn),key=lambda r:r[0]):
        print(f"{int(tt//60)}:{tt%60:05.2f}\t{tt:.2f}\t{ff:.0f}")
