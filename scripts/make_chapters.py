#!/usr/bin/env python3
"""検出した笛の時刻を mov にチャプターマーカーとして埋め込む。

results/whistles.txt を読み、results/chapters.ffmeta を生成し、
ffmpeg のストリームコピー(再エンコードなし)で
results/2026-l1-final-1st_marked.mov を作成する。

使い方:  python3 make_chapters.py
"""
import subprocess, os

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_MOV  = os.path.join(ROOT, "data",    "2026-l1-final-1st.mov")
WHISTLES = os.path.join(ROOT, "results", "whistles.txt")
FFMETA   = os.path.join(ROOT, "results", "chapters.ffmeta")
OUT_MOV  = os.path.join(ROOT, "results", "2026-l1-final-1st_marked.mov")

def mov_duration(path):
    out = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                          "-of","default=nk=1:nw=1", path], capture_output=True, text=True)
    return float(out.stdout.strip())

def main():
    rows = [l.split("\t") for l in open(WHISTLES) if not l.startswith("#") and l.strip()]
    times = [float(r[1]) for r in rows]
    end = mov_duration(SRC_MOV)
    with open(FFMETA, "w") as f:
        f.write(";FFMETADATA1\n")
        # 先頭に開始チャプター(0->笛01)。これで笛01が正確な時刻から始まる
        f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND={int(round(times[0]*1000))-1}\ntitle=開始\n")
        for i, t in enumerate(times):
            start = int(round(t*1000))
            stop  = int(round((times[i+1] if i+1 < len(times) else end)*1000)) - 1
            m, s = int(t//60), t % 60
            f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={stop}\ntitle=笛{i+1:02d} ({m}:{s:05.2f})\n")
    print(f"{len(times)} 笛チャプター + 開始 を {os.path.relpath(FFMETA, ROOT)} に生成")

    subprocess.run(["ffmpeg","-v","error","-y","-i",SRC_MOV,"-i",FFMETA,
                    "-map_metadata","1","-map_chapters","1","-c","copy", OUT_MOV], check=True)
    print(f"作成: {os.path.relpath(OUT_MOV, ROOT)}")

if __name__ == "__main__":
    main()
