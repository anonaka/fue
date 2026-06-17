#!/usr/bin/env python3
"""教師データ作成支援ツール: 候補レビュー用HTMLを生成する。

検出器(アンサンブル)の候補を合議スコア順に並べ、各候補について
  ・±2.5秒の音声クリップ(その場で再生)
  ・スペクトログラム画像
を埋め込んだ自己完結HTMLを出力。ブラウザでキーボード操作(w=笛/n=非笛/u=保留)で
高速にラベル付けし、labels/ground_truth.tsv 形式でエクスポートできる。

使い方: python3 make_review.py [in.wav] [N] [out.html]
        N=レビューする候補数(既定=全部), out=出力HTML(既定 results/review.html)
"""
import os, sys, base64, subprocess, html
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_whistle as D

CLIP=2.5; SPEC_W,SPEC_H=380,200

def clip_mp3(path,ts):
    ss=max(0,ts-CLIP/2)
    return subprocess.run(["ffmpeg","-v","error","-ss",str(ss),"-t",str(CLIP),"-i",path,
        "-ac","1","-c:a","libmp3lame","-b:a","64k","-f","mp3","-"],capture_output=True).stdout

def spec_jpg(path,ts):
    ss=max(0,ts-CLIP/2)
    return subprocess.run(["ffmpeg","-v","error","-ss",str(ss),"-t",str(CLIP),"-i",path,
        "-lavfi",f"showspectrumpic=s={SPEC_W}x{SPEC_H}:legend=0:fscale=lin:stop=6000:color=intensity",
        "-c:v","mjpeg","-q:v","5","-f","image2","-"],capture_output=True).stdout

def main(path,N,out):
    cands=D.candidates(path)
    if N: cands=cands[:N]
    print(f"{len(cands)} 候補をレンダリング中...",file=sys.stderr)
    cards=[]; js=[]
    for i,(t,f0,sc,src) in enumerate(cands):
        mp3=base64.b64encode(clip_mp3(path,t)).decode()
        jpg=base64.b64encode(spec_jpg(path,t)).decode()
        mmss=f"{int(t//60)}:{t%60:05.2f}"; srcs="+".join(sorted(src))
        cards.append(f'''<div class="card" id="c{i}" data-i="{i}">
<div class="hd"><span class="num">#{i+1}</span> <b>{mmss}</b>
 <span class="meta">f0={f0:.0f} score={sc} [{srcs}]</span> <span class="lab" id="l{i}"></span></div>
<img loading="lazy" src="data:image/jpeg;base64,{jpg}">
<audio controls preload="none" src="data:audio/mpeg;base64,{mp3}"></audio></div>''')
        js.append(f'{{t:{t:.2f},mmss:"{mmss}",f0:"{f0:.0f}"}}')
        if (i+1)%20==0: print(f"  {i+1}/{len(cands)}",file=sys.stderr)
    htmlpage=TEMPLATE.replace("/*CARDS*/","\n".join(cards)).replace("/*DATA*/",",".join(js))
    open(out,"w").write(htmlpage)
    print(f"出力: {out}  ({os.path.getsize(out)//1024} KB)",file=sys.stderr)
    print(f"ブラウザで開く:  open {out}",file=sys.stderr)

TEMPLATE=r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>笛ラベル付け</title><style>
body{font-family:-apple-system,sans-serif;margin:0;background:#1a1a1a;color:#eee}
header{position:sticky;top:0;background:#222;padding:10px 16px;border-bottom:1px solid #444;z-index:9}
header b{color:#6cf}.btn{background:#357;color:#fff;border:0;padding:6px 12px;border-radius:5px;cursor:pointer;margin-left:6px}
.hint{font-size:12px;color:#aaa;margin-top:4px}
.wrap{max-width:760px;margin:12px auto;padding:0 12px}
.card{background:#2a2a2a;border:2px solid #333;border-radius:8px;padding:10px;margin:10px 0}
.card.cur{border-color:#6cf}.card.w{border-color:#3c6;background:#1e2c20}.card.n{border-color:#c55;background:#2c1e1e}
.card.u{border-color:#cc5;background:#2c2c1e}
.hd{display:flex;align-items:center;gap:8px;margin-bottom:6px}.num{color:#888}.meta{font-size:12px;color:#9ab}
.lab{margin-left:auto;font-weight:bold}img{width:100%;border-radius:4px;display:block}
audio{width:100%;margin-top:8px}
</style></head><body>
<header><span id="prog"></span>
<button class="btn" onclick="exp()">エクスポート(TSV)</button>
<button class="btn" onclick="cp()">クリップボードにコピー</button>
<div class="hint">↓/j 次・↑/k 前・Space 再生・<b style="color:#3c6">W=笛</b>・<b style="color:#c55">N=非笛</b>・<b style="color:#cc5">U=保留</b>(ラベルで自動的に次へ)</div>
</header><div class="wrap">/*CARDS*/</div>
<script>
const C=[/*DATA*/]; const L={}; let cur=0;
function paint(i){const c=document.getElementById('c'+i);c.classList.remove('w','n','u');
 const m={whistle:'w',not_whistle:'n',unsure:'u'};const t={whistle:'🟢 笛',not_whistle:'🔴 非笛',unsure:'🟡 保留'};
 if(L[i]){c.classList.add(m[L[i]]);document.getElementById('l'+i).textContent=t[L[i]];}
 else document.getElementById('l'+i).textContent='';}
function focusCard(i){document.querySelectorAll('.card').forEach(e=>e.classList.remove('cur'));
 const c=document.getElementById('c'+i);if(c){c.classList.add('cur');c.scrollIntoView({block:'center',behavior:'smooth'});}cur=i;prog();}
function setL(i,v){L[i]=v;paint(i);prog();if(i+1<C.length)focusCard(i+1);}
function prog(){const n=Object.keys(L).length;document.getElementById('prog').innerHTML=
 '<b>'+n+'</b> / '+C.length+' ラベル済み (#'+(cur+1)+')';}
function play(i){const a=document.getElementById('c'+i).querySelector('audio');a.currentTime=0;a.play();}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
 const k=e.key.toLowerCase();
 if(k==='arrowdown'||k==='j'){e.preventDefault();focusCard(Math.min(cur+1,C.length-1));}
 else if(k==='arrowup'||k==='k'){e.preventDefault();focusCard(Math.max(cur-1,0));}
 else if(k===' '){e.preventDefault();play(cur);}
 else if(k==='w'){setL(cur,'whistle');} else if(k==='n'){setL(cur,'not_whistle');}
 else if(k==='u'){setL(cur,'unsure');}});
document.querySelectorAll('.card').forEach(c=>c.addEventListener('click',()=>focusCard(+c.dataset.i)));
function tsv(){let o='# mm:ss\tseconds\tf0(Hz)\tlabel\tverified\tnote\n';
 C.forEach((c,i)=>{if(L[i]&&L[i]!=='unsure')o+=`${c.mmss}\t${c.t.toFixed(2)}\t${c.f0}\t${L[i]}\thuman_ear\treview\n`;});return o;}
function exp(){const b=new Blob([tsv()],{type:'text/tab-separated-values'});const a=document.createElement('a');
 a.href=URL.createObjectURL(b);a.download='labels_review.tsv';a.click();}
function cp(){navigator.clipboard.writeText(tsv()).then(()=>alert('コピーしました'));}
focusCard(0);prog();
</script></body></html>"""

if __name__=="__main__":
    path=sys.argv[1] if len(sys.argv)>1 else os.path.join(ROOT,"data","2026-l1-final-1st.wav")
    N=int(sys.argv[2]) if len(sys.argv)>2 and sys.argv[2]!="0" else None
    out=sys.argv[3] if len(sys.argv)>3 else os.path.join(ROOT,"results","review.html")
    main(path,N,out)
