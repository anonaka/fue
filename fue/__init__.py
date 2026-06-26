"""笛検出の呼び出し可能なAPI（検出器は切り替え可能）。

検出器を選んで外部から import して使える1つの手続きとして公開する。

    from fue import detect, DETECTORS
    detect("/path/to/video.mp4")                      # 既定の検出器
    detect("/path/to/video.mp4", detector="ensemble") # v1∪v2(放送向け)
    detect("/path/to/video.mp4", detector="v3")       # 横リッジ(アマチュア向け)

検出器:
    "v3"       横リッジ方式 (scripts/detect_v3.py)。家庭用ビデオ録画で特に有効。既定。
    "ensemble" v1∪v2 アンサンブル (scripts/detect_whistle.py)。放送映像向け。

入力 `url` は ffmpeg がデコードできるもの（動画/音声のパスまたはURL）。
戻り値は検出された笛の `[{"timestamp": 秒, "confidence": 0..1}, ...]`（timestamp昇順）。
confidence は検出スコアをそのランの p95 で正規化した値（信号強度に比例。max でなく p95
正規化なので外れ値1個に引きずられず、検出器・録音をまたいで min_confidence が一貫する）。
"""
import os
import sys

import numpy as np

# 既存スクリプト (scripts/) を import できるよう sys.path に追加する
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import detect_whistle as _ensemble  # noqa: E402  (sys.path 調整後に import)
import detect_whistle_v1 as _v1  # noqa: E402
import detect_whistle_v2 as _v2  # noqa: E402
import detect_v3 as _v3  # noqa: E402

DEFAULT_DETECTOR = os.environ.get("FUE_DETECTOR", "v3")

__all__ = ["detect", "DETECTORS", "DEFAULT_DETECTOR"]


def _run_ensemble(url, progress):
    """v1∪v2 を実行し [(time, f0, score), ...] を返す。進捗: v1 0..45 / v2 45..90。"""
    if progress is not None:
        _v1.PROGRESS = lambda i, n: progress(int(i / max(n, 1) * 45))
        _v2.PROGRESS = lambda i, n: progress(45 + int(i / max(n, 1) * 45))
    try:
        rows = _ensemble.candidates(url)   # [(time, f0, score, sources_set), ...]
    finally:
        _v1.PROGRESS = None
        _v2.PROGRESS = None
    return [(t, f0, score) for t, f0, score, _src in rows]


def _run_v3(url, progress):
    """v3(横リッジ)を実行し [(time, f0, score), ...] を返す。進捗: スペクトログラム 0..90。"""
    if progress is not None:
        _v3.PROGRESS = lambda i, n: progress(int(i / max(n, 1) * 90))
    try:
        rows = _v3.candidates(url)         # [(time, f0, score), ...]
    finally:
        _v3.PROGRESS = None
    return rows


# 検出器レジストリ（名前 -> 実行関数）。新しい検出器はここに追加するだけで切り替え可能。
DETECTORS = {"v3": _run_v3, "ensemble": _run_ensemble}


def detect(url, *, min_confidence=0.0, progress=None, detector=None):
    """笛を検出して [{"timestamp": 秒(float), "confidence": 0..1(float)}, ...] を返す。

    detector で検出器を選ぶ（既定は環境変数 FUE_DETECTOR か "v3"）。DETECTORS のキー参照。
    min_confidence 未満の候補は除外する。
    progress に func(percent:int) を渡すと、解析の進捗(0..100)を通知する。
    """
    name = detector or DEFAULT_DETECTOR
    if name not in DETECTORS:
        raise ValueError(f"unknown detector {name!r} (available: {', '.join(DETECTORS)})")
    rows = DETECTORS[name](url, progress)
    if progress is not None:
        progress(95)
    if not rows:
        if progress is not None:
            progress(100)
        return []
    norm = float(np.percentile([r[2] for r in rows], 95)) or 1.0
    out = []
    for t, _f0, score in rows:
        conf = min(score / norm, 1.0)
        if conf >= min_confidence:
            out.append({"timestamp": round(float(t), 2), "confidence": round(float(conf), 4)})
    out.sort(key=lambda r: r["timestamp"])
    if progress is not None:
        progress(100)
    return out
