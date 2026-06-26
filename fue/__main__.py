"""`python -m fue <url> [min_confidence] [detector]` — detect() の結果をJSONでstdout出力。

Node など他プロセスから呼び出すための薄いエントリ。検出結果以外は出力しない
（ログ・進捗は stderr）。detector 省略時は環境変数 FUE_DETECTOR か既定の検出器。
"""
import json
import sys

from fue import detect, DETECTORS


def main():
    if len(sys.argv) < 2:
        print(f"usage: python -m fue <url> [min_confidence] [{'|'.join(DETECTORS)}]", file=sys.stderr)
        sys.exit(2)
    url = sys.argv[1]
    min_confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    detector = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None

    # 解析の進捗は stderr に "progress:NN" 形式で出す（呼び出し側が拾う）。
    def report(pct):
        print(f"progress:{pct}", file=sys.stderr, flush=True)

    result = detect(url, min_confidence=min_confidence, progress=report, detector=detector)
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
