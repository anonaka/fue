#!/usr/bin/env python3
"""レビューHTMLを配信し、ラベルを自動保存するローカルサーバ。

ブラウザがラベル変更のたびに /save へPOSTするので、エクスポート操作は不要。
保存先ファイルは常に最新のラベルに更新される。

使い方: python3 review_server.py <review.html> <save.tsv> [port]
        ブラウザで表示されたURL(http://localhost:PORT)を開く。
"""
import sys, os, http.server

def make_handler(html_bytes, save_path):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8")
                self.send_header("Content-Length",str(len(html_bytes))); self.end_headers()
                self.wfile.write(html_bytes)
            else:
                self.send_response(404); self.end_headers()
        def do_POST(self):
            if self.path == "/save":
                n=int(self.headers.get("Content-Length",0)); body=self.rfile.read(n)
                with open(save_path,"wb") as f: f.write(body)
                cnt=sum(1 for l in body.decode("utf-8","ignore").splitlines()
                        if l and not l.startswith("#"))
                print(f"  保存 {cnt}件 -> {save_path}", flush=True)
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
            else:
                self.send_response(404); self.end_headers()
    return H

def main(html, save, port):
    html_bytes=open(html,"rb").read()
    srv=http.server.HTTPServer(("127.0.0.1",port), make_handler(html_bytes, save))
    print(f"配信中: http://localhost:{port}  (ラベルは自動保存 -> {save})")
    print("ブラウザで上記URLを開いてラベル付け。終了は Ctrl-C。")
    try: srv.serve_forever()
    except KeyboardInterrupt: print("\n停止")

if __name__=="__main__":
    if len(sys.argv)<3:
        print("usage: review_server.py <review.html> <save.tsv> [port]"); sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 8765)
