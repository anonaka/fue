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
confidence は、学習済み分類器(models/whistle_clf.npz)があれば whistle probability(0..1)、
無ければ検出スコアの p95 正規化値(外れ値に強く、検出器・録音をまたいで min_confidence が一貫)。
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


def _model_path():
    """学習済み分類器の npz パス(あれば)。env FUE_WHISTLE_MODEL か <fueルート>/models/whistle_clf.npz。"""
    p = os.environ.get("FUE_WHISTLE_MODEL")
    if p and os.path.exists(p):
        return p
    default = os.path.join(os.path.dirname(_SCRIPTS), "models", "whistle_clf.npz")
    return default if os.path.exists(default) else None


def _run_ensemble(url, progress):
    """v1∪v2。戻り値 (rows[(t,f0,score)], is_prob=False)。"""
    if progress is not None:
        _v1.PROGRESS = lambda i, n: progress(int(i / max(n, 1) * 45))
        _v2.PROGRESS = lambda i, n: progress(45 + int(i / max(n, 1) * 45))
    try:
        rows = _ensemble.candidates(url)   # [(time, f0, score, sources_set), ...]
    finally:
        _v1.PROGRESS = None
        _v2.PROGRESS = None
    return [(t, f0, score) for t, f0, score, _src in rows], False


def _run_v3(url, progress):
    """v3(横リッジ+ステレオ)。学習済み分類器があれば whistle probability で再ランクする。
    戻り値 (rows[(t,f0,value)], is_prob)。value は分類器の確率 or ridge スコア。"""
    if progress is not None:
        _v3.PROGRESS = lambda i, n: progress(int(i / max(n, 1) * 90))
    mp = _model_path()
    model = _v3.load_model(mp) if mp else None
    try:
        rows = _v3.detect_scored(url, model)
    finally:
        _v3.PROGRESS = None
    return rows, (model is not None)


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
    rows, is_prob = DETECTORS[name](url, progress)
    if progress is not None:
        progress(95)
    if not rows:
        if progress is not None:
            progress(100)
        return []
    if is_prob:                                    # 分類器の確率はそのまま confidence
        conf_of = lambda v: min(max(v, 0.0), 1.0)
    else:                                          # 検出スコアは p95 正規化
        norm = float(np.percentile([r[2] for r in rows], 95)) or 1.0
        conf_of = lambda v: min(v / norm, 1.0)
    out = []
    for t, _f0, val in rows:
        conf = conf_of(val)
        if conf >= min_confidence:
            out.append({"timestamp": round(float(t), 2), "confidence": round(float(conf), 4)})
    out.sort(key=lambda r: r["timestamp"])
    if progress is not None:
        progress(100)
    return out
