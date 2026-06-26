#!/usr/bin/env python3
"""検出した笛の時刻を mov にチャプターマーカーとして埋め込む。

ラベルTSV(whistle行)の時刻を読み、chapters_<stem>.ffmeta を生成し、
ffmpeg のストリームコピー(再エンコードなし)でマーカー付き mov を作成する。
出力(ffmeta / marked.mov)は out.mov と同じフォルダに置かれる。

使い方:
  python3 make_chapters.py <src.mov> <out.mov> <labels1.tsv> [labels2.tsv ...]
"""
import subprocess, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_times(path):
    """秒(2列目)を返す。label列(4列目)がある場合は whistle のみ採用。
    not_whistle 等の負例はマーカーにしない(教師データには残す)。"""
    if not os.path.exists(path): return []
    out = []
    for l in open(path):
        if l.startswith("#") or not l.strip(): continue
        cols = l.split("\t")
        if len(cols) >= 4 and cols[3].strip() != "whistle": continue
        out.append(float(cols[1]))
    return out

def mov_duration(path):
    out = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                          "-of","default=nk=1:nw=1", path], capture_output=True, text=True)
    return float(out.stdout.strip())

def make(src_mov, out_mov, label_files):
    times = []
    for lf in label_files: times += read_times(lf)
    times.sort()
    merged = []   # 近接(1秒以内)の重複除去。連続した別々の笛(例16:26/16:27)は残す
    for t in times:
        if not merged or t - merged[-1] >= 1.0: merged.append(t)
    times = merged
    end = mov_duration(src_mov)
    base = os.path.splitext(os.path.basename(out_mov))[0]
    if base.endswith("_marked"): base = base[:-len("_marked")]
    ffmeta = os.path.join(os.path.dirname(out_mov), "chapters_"+base+".ffmeta")
    with open(ffmeta, "w") as f:
        f.write(";FFMETADATA1\n")
        f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND={int(round(times[0]*1000))-1}\ntitle=開始\n")
        for i, t in enumerate(times):
            start = int(round(t*1000))
            stop  = int(round((times[i+1] if i+1 < len(times) else end)*1000)) - 1
            m, s = int(t//60), t % 60
            f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={stop}\ntitle=笛{i+1:02d} ({m}:{s:05.2f})\n")
    print(f"{len(times)} 笛チャプター + 開始 を {os.path.relpath(ffmeta, ROOT)} に生成")
    subprocess.run(["ffmpeg","-v","error","-y","-i",src_mov,"-i",ffmeta,
                    "-map_metadata","1","-map_chapters","1","-c","copy", out_mov], check=True)
    print(f"作成: {os.path.relpath(out_mov, ROOT)}")

if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 3:
        sys.exit("使い方: python3 make_chapters.py <src.mov> <out.mov> <labels1.tsv> [labels2.tsv ...]")
    make(a[0], a[1], a[2:])
