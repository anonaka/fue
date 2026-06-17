#!/usr/bin/env python3
"""HPSS(調波・打楽器分離)前処理。

スペクトログラムを
  時間方向メディアン -> 持続トーン(調波=笛・声) を抽出
  周波数方向メディアン -> 瞬発音(打楽器=ボール音・拍手) を抽出
し、ソフトマスクで調波成分を強調(打楽器・広帯域ノイズを抑制)した音声を書き出す。
検出器の内部仮定(低域の群衆床など)を壊さないよう、ハードなバンドパスはしない。

使い方: python3 preprocess.py [in.wav] [out.wav]
"""
import subprocess, sys, os, numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SR, NFFT, HOP = 16000, 1024, 512
KT, KF = 7, 9   # 時間/周波数メディアンのカーネル長

def load(path):
    raw=subprocess.run(["ffmpeg","-v","error","-i",path,"-ac","1","-ar",str(SR),
                        "-f","s16le","-"],capture_output=True).stdout
    return np.frombuffer(raw,dtype="<i2").astype(np.float32)/32768.0

def med_time(mag,k):
    pad=k//2; out=np.empty_like(mag)
    for b in range(mag.shape[1]):
        col=np.pad(mag[:,b],(pad,pad),mode="edge")
        out[:,b]=np.median(sliding_window_view(col,k),axis=1)
    return out

def med_freq(mag,k):
    pad=k//2; M=np.pad(mag,((0,0),(pad,pad)),mode="edge"); out=np.empty_like(mag)
    for s in range(0,mag.shape[0],4000):
        e=min(s+4000,mag.shape[0])
        out[s:e]=np.median(sliding_window_view(M[s:e],k,axis=1),axis=2)
    return out

def main(inp,outp):
    x=load(inp); win=np.hanning(NFFT).astype(np.float32)
    frames=sliding_window_view(x,NFFT)[::HOP]*win
    S=np.fft.rfft(frames,axis=1); mag=np.abs(S).astype(np.float32)
    print("frames",S.shape, file=sys.stderr)
    H=med_time(mag,KT); P=med_freq(mag,KF)
    mask=(H*H)/(H*H+P*P+1e-9)            # 調波ソフトマスク
    Sout=S*mask
    rec=np.fft.irfft(Sout,n=NFFT,axis=1)*win   # (nf,NFFT)
    y=np.zeros(len(x),np.float32); wsum=np.zeros(len(x),np.float32); w2=win*win
    for i in range(rec.shape[0]):
        a=i*HOP; y[a:a+NFFT]+=rec[i]; wsum[a:a+NFFT]+=w2
    y/=np.maximum(wsum,1e-6)
    y=np.clip(y/ (np.abs(y).max()+1e-9)*0.97, -1,1)
    pcm=(y*32767).astype("<i2").tobytes()
    subprocess.run(["ffmpeg","-v","error","-y","-f","s16le","-ar",str(SR),"-ac","1",
                    "-i","-","-ar",str(SR),outp],input=pcm,check=True)
    print("wrote",outp,file=sys.stderr)

if __name__=="__main__":
    inp=sys.argv[1] if len(sys.argv)>1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data","2026-l1-final-1st.wav")
    outp=sys.argv[2] if len(sys.argv)>2 else "/tmp/preproc_hpss.wav"
    main(inp,outp)
