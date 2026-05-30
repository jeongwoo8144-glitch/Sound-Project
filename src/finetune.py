"""End-to-end YAMNet fine-tuning for ADAS sound detection.
(ADAS 음향 탐지를 위한 YAMNet 전체 파인튜닝)

Step 4b — optional improvement after Step 4 (classifier.py).
(Step 4 분류기 학습 후 선택적으로 실행하는 개선 단계)

[ 파인튜닝 전략 ]
* models/custom_classifier.h5에서 사전 학습된 분류기 가중치 로드.
* TF-Hub에서 YAMNet 로드 (hub.load).
* YAMNet 변수의 앞 80%(초기 DSC 블록 = 일반 특징)는 동결.
* 마지막 20%(고수준 표현)만 yamnet_lr=1e-5로 파인튜닝.
* 분류기 헤드는 clf_lr=1e-4 (이미 잘 학습된 헤드에는 더 높은 lr 허용).
* 수동 GradientTape 루프 (watch_accessed_variables=False)
  → 동결된 YAMNet 변수에 기울기가 쌓이지 않아 메모리 절약.
* 샘플별 클래스 가중치 + 양성 클래스 boost (classifier.py와 동일).
* Early stopping on val_loss (patience=5).

[ 왜 GradientTape를 수동으로 사용하는가? ]
TF Hub의 hub.load()는 _UserObject를 반환하며,
이 객체에는 .trainable_variables가 없습니다.
따라서 model.fit()처럼 자동으로 기울기를 계산할 수 없고,
더미 입력으로 전진 pass를 실행한 뒤 tape.watched_variables()로
YAMNet의 56개 변수를 직접 발견해야 합니다.

[ 저장 아티팩트 ]
  models/yamnet_finetuned/          ← TF 체크포인트 (hub 모델에 복원)
  models/custom_classifier_finetuned.h5  ← 파인튜닝된 분류기

[ 추론 ]
  realtime_infer.py의 YAMNetEngine이 yamnet.finetuned_dir을
  config.yaml에서 설정했을 때 자동으로 파인튜닝 체크포인트를 로드합니다.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Windows에서 sklearn보다 TF를 먼저 import해야 DLL 충돌 방지
import tensorflow as tf  # type: ignore[import]

from .utils.config import get_class_map, load_config, setup_logging
from .classifier import (
    build_label_map,
    fold_split,
    get_class_weights,
    evaluate_model,
)


# ---------------------------------------------------------------------------
# 파형(Waveform) 데이터셋 처리
# ---------------------------------------------------------------------------

def _resolve_to_urbansound(
    wav_path: Path,
    project_root: Path,
) -> tuple[Path, int]:
    """삭제된 처리 클립 경로를 원본 소스 파일로 역추적합니다.

    [ 두 가지 소스 유형 처리 ]

    1. UrbanSound8K — 파일명에 슬라이스 인덱스 포함:
       ``{freesound_id}-{class_id}-{take}-{slice}.wav``  (예: ``102853-8-0-0.wav``)
       처리된 경로 → 원본:
           data/processed/{variant}/fold{N}/{class}/{filename}
           → data/raw/UrbanSound8K/audio/fold{N}/{filename}

    2. STRAFFIC — 수 분짜리 채널 녹음을 청크로 분할:
       ``ch01_clip0000.wav`` → 채널 ``ch01.wav``, 청크 인덱스 0
       처리된 경로 → 원본:
           data/processed/{variant}/fold{N}/{class}/ch{XX}_clip{NNNN}.wav
           → data/raw/straffic/ch{XX}.wav  (chunk_idx = NNNN)

    Returns:
        (resolved_path, chunk_idx) — chunk_idx는 소스 파일의 4초 오프셋
        (UrbanSound8K: 0, STRAFFIC: N번째 청크)
    """
    import re

    parts = wav_path.parts
    try:
        proc_idx = parts.index("processed")
    except ValueError:
        return wav_path, 0

    # parts 구조: [..., 'processed', variant, fold_dir, class_name, filename]
    if len(parts) < proc_idx + 5:
        return wav_path, 0

    fold_dir = parts[proc_idx + 2]   # 예: 'fold7'
    filename = parts[proc_idx + 4]   # 예: '102853-8-0-0.wav' or 'ch01_clip0000.wav'

    # ── STRAFFIC 패턴: ch{XX}_clip{NNNN}.wav ──
    straffic_match = re.match(r"^(ch\d+)_clip(\d+)\.wav$", filename)
    if straffic_match:
        channel = straffic_match.group(1)        # 'ch01'
        chunk_idx = int(straffic_match.group(2)) # 0, 1, 2, …
        straffic_path = project_root / "data" / "raw" / "straffic" / f"{channel}.wav"
        return straffic_path, chunk_idx

    # ── UrbanSound8K: 파일명이 바로 클립, 추가 청킹 없음 ──
    orig_path = (
        project_root / "data" / "raw" / "UrbanSound8K"
        / "audio" / fold_dir / filename
    )
    return orig_path, 0


def _dedup_to_clean(df: pd.DataFrame) -> pd.DataFrame:
    """'clean' 변형 행만 유지하고 SNR 증강 중복을 제거합니다.

    [ 이유 ]
    처리된 클립이 없으면 UrbanSound8K 원본으로 폴백하는데,
    SNR 변형들을 로드하면 동일한 파형을 5번씩 읽는 불필요한 중복이 발생합니다.
    clean 서브셋으로 중복 제거하면 STRAFFIC 행도 보존됩니다.
    """
    path_col = df["path"].str.replace("\\", "/", regex=False)
    is_snr = path_col.str.contains(r"processed/snr_", regex=False)
    return df[~is_snr].copy()


def load_waveforms(
    df: pd.DataFrame,
    project_root: Path,
    sample_rate: int,
    target_samples: int,
    label_map: dict[int, int],
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """df의 모든 파형을 (N, target_samples) float32 배열로 로드합니다.

    [ 폴백 메커니즘 ]
    처리된 클립 파일이 없으면 (예: 정리 후) 자동으로 원본 UrbanSound8K 파일로
    폴백하여 올바른 청크를 오프셋으로 추출합니다.

    [ YAMNet 요구사항 ]
    sample_rate: 16000 Hz (YAMNet 고정)
    target_samples: 64000 (= 16000 × 4초)

    Args:
        df: manifest DataFrame의 서브셋
        project_root: 절대 프로젝트 루트 (df의 경로는 상대 경로)
        sample_rate: 목표 샘플레이트 (YAMNet = 16000)
        target_samples: 패드/절단 후 파형 길이 (64000 = 4초)
        label_map: {class_id → dense_label}
        logger: 로깅 인스턴스

    Returns:
        (X, y) — X: (N, target_samples), y: (N,) dense 정수 레이블
    """
    import re
    import librosa  # type: ignore[import]

    # SNR 증강 중복 제거 (clean만 유지)
    df = _dedup_to_clean(df)
    logger.info("  After dedup (clean-only): %d rows", len(df))

    _STRAFFIC_RE = re.compile(r"^(ch\d+)_clip(\d+)\.wav$")

    # STRAFFIC 채널 파일 사전 캐싱: 채널 파일 1개를 한 번만 로드
    # (300초짜리 파일을 4초 클립마다 다시 여는 비효율 방지)
    straffic_cache: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        filename = Path(row["path"].replace("\\", "/")).name
        m = _STRAFFIC_RE.match(filename)
        if m:
            channel = m.group(1)
            if channel not in straffic_cache:
                channel_path = (
                    project_root / "data" / "raw" / "straffic" / f"{channel}.wav"
                )
                if channel_path.exists():
                    try:
                        full_audio, _ = librosa.load(
                            str(channel_path), sr=sample_rate, mono=True
                        )
                        straffic_cache[channel] = full_audio
                        logger.info(
                            "  Pre-cached straffic %s — %d samples (%.1f s)",
                            channel, len(full_audio), len(full_audio) / sample_rate,
                        )
                    except Exception as exc:
                        logger.warning("Failed to pre-cache %s: %s", channel_path, exc)
                else:
                    logger.warning("Straffic source not found: %s", channel_path)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    n = len(df)
    missing = 0
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            logger.info("  Loading waveforms … %d / %d", i, n)

        row_path = row["path"].replace("\\", "/")
        filename = Path(row_path).name
        straffic_match = _STRAFFIC_RE.match(filename)

        if straffic_match:
            # STRAFFIC 경로: 사전 캐싱된 채널 배열에서 슬라이스
            channel = straffic_match.group(1)
            chunk_idx = int(straffic_match.group(2))
            if channel not in straffic_cache:
                missing += 1
                if missing <= 5:
                    logger.warning(
                        "Straffic channel not cached (skipping): %s", channel
                    )
                continue
            start = chunk_idx * target_samples
            audio = straffic_cache[channel][start : start + target_samples]
        else:
            # UrbanSound8K (또는 기타): 개별 파일 로드, 없으면 폴백
            wav_path = project_root / row_path
            chunk_idx = 0

            if not wav_path.exists():
                # 처리된 클립이 없으면 원본 파일로 폴백
                wav_path, chunk_idx = _resolve_to_urbansound(wav_path, project_root)

            if not wav_path.exists():
                missing += 1
                if missing <= 5:
                    logger.warning("File not found (skipping): %s", wav_path)
                continue

            try:
                audio, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", wav_path, exc)
                continue

            # STRAFFIC 청크 오프셋 적용
            start = chunk_idx * target_samples
            audio = audio[start : start + target_samples]

        # 패드(짧으면 0으로 채움) 또는 절단(길면 자름)
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))

        X_list.append(audio.astype(np.float32))
        y_list.append(int(label_map[int(row["class_id"])]))

    if missing > 5:
        logger.warning("… and %d more missing files (skipped total: %d)", missing - 5, missing)

    if not X_list:
        raise RuntimeError("No waveforms loaded — check data paths in manifest.csv")

    X = np.stack(X_list)  # (N, target_samples) 배열로 쌓기
    y = np.array(y_list, dtype=np.int32)
    logger.info("Loaded %d waveforms — shape %s", len(X), X.shape)
    return X, y


def make_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """numpy 배열을 배치된 tf.data.Dataset으로 변환합니다.

    prefetch(AUTOTUNE): CPU가 GPU 학습과 병렬로 다음 배치를 준비 → 처리량 향상
    shuffle: 매 에폭 데이터 순서를 섞어 과적합 방지 (train에서만 True)
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# YAMNet 부분 동결 헬퍼
# ---------------------------------------------------------------------------

def discover_yamnet_vars(
    yamnet: Any,
    logger: logging.Logger,
) -> list[tf.Variable]:
    """더미 GradientTape 전진 pass로 YAMNet의 모든 변수를 발견합니다.

    [ 왜 이런 방법이 필요한가? ]
    hub.load()가 반환하는 _UserObject에는 .trainable_variables가 없습니다.
    일반적인 model.trainable_variables로 접근이 불가능하므로,
    더미 입력을 넣어 전진 pass를 실행하면 TF가 접근된 모든 변수를 추적합니다.
    tape.watched_variables()로 그 목록을 얻을 수 있습니다.

    [ 정렬의 중요성 ]
    변수를 이름 기준으로 정렬하면 실행마다 순서가 동일하게 보장됩니다.
    이를 통해 "마지막 60%"를 안정적으로 선택할 수 있습니다.

    Args:
        yamnet: hub.load()로 로드된 TF-Hub 모듈
        logger: 로깅 인스턴스

    Returns:
        이름 기준으로 정렬된 tf.Variable 목록 (총 56개)
    """
    # YAMNet은 1초당 15600 샘플 (0.975초 × 16000) 단위로 처리
    dummy = tf.zeros(15600, dtype=tf.float32)
    with tf.GradientTape() as tape:
        # 전진 pass: yamnet은 (log-mel 스펙트로그램, 임베딩, 프레임당 예측) 반환
        _, emb, _ = yamnet(dummy)
        _ = tf.reduce_sum(emb)
    # tape이 기억한 모든 변수를 이름 기준 정렬
    all_vars = sorted(tape.watched_variables(), key=lambda v: v.name)
    logger.info("YAMNet discovered %d variables via GradientTape.", len(all_vars))
    return all_vars


def select_yamnet_trainable_vars(
    all_yamnet_vars: list[tf.Variable],
    unfreeze_fraction: float,
    logger: logging.Logger,
) -> list[tf.Variable]:
    """YAMNet 발견 변수 중 마지막 unfreeze_fraction 비율만 반환합니다.

    [ 전략 ]
    변수를 이름(레이어 순서)으로 정렬 후 마지막 N%만 학습 가능으로 설정합니다.
    - 초기 레이어 (Block 1-8): 동결 → AudioSet에서 학습된 일반 특징 보존
    - 후기 레이어 (Block 9-14): 학습 가능 → ADAS 도메인 고수준 표현 적응

    Phase 4 설정: unfreeze_fraction=0.60 → 56개 중 약 33개 변수 학습

    Args:
        all_yamnet_vars: discover_yamnet_vars()의 전체 변수 목록
        unfreeze_fraction: 해제 비율 (예: 0.60 = 마지막 60%)
        logger: 로깅 인스턴스

    Returns:
        기울기 업데이트를 받을 변수 서브 리스트
    """
    n_total = len(all_yamnet_vars)
    n_unfreeze = max(1, int(n_total * unfreeze_fraction))
    vars_to_train = all_yamnet_vars[-n_unfreeze:]  # 마지막 N개 선택

    logger.info(
        "YAMNet variables — total: %d  frozen: %d  trainable: %d  (%.0f%%)",
        n_total, n_total - n_unfreeze, n_unfreeze, unfreeze_fraction * 100,
    )
    logger.info(
        "Trainable YAMNet vars: %s … %s",
        vars_to_train[0].name, vars_to_train[-1].name,
    )
    return vars_to_train


# ---------------------------------------------------------------------------
# 학습 스텝 함수
# ---------------------------------------------------------------------------

def make_train_step(
    yamnet: Any,
    classifier: tf.keras.Model,
    yamnet_vars_to_train: list[tf.Variable],
    yamnet_optimizer: tf.keras.optimizers.Optimizer,
    clf_optimizer: tf.keras.optimizers.Optimizer,
    weight_vector: tf.Tensor,
    use_sigmoid: bool = False,
) -> Any:
    """컴파일된 tf.function 학습 스텝을 반환합니다.

    [ 핵심 설계 ]
    기본 GradientTape (watch_accessed_variables=True)를 사용하여
    YAMNet의 내부 변수들(hub-load된 _UserObject로 직접 접근 불가)이
    전진 pass 중 자동으로 추적됩니다.
    그 후 yamnet_vars_to_train(마지막 N%)과 분류기 변수에 대해서만
    기울기를 요청합니다. 나머지는 추적되지만 업데이트 받지 않습니다.

    [ 이중 옵티마이저 ]
    yamnet_optimizer: yamnet_lr=5e-6 (매우 작은 학습률 → 기존 특징 최소 변형)
    clf_optimizer:    clf_lr=2e-5   (분류기 헤드는 좀 더 빠르게 적응)

    Args:
        use_sigmoid: True이면 BinaryCrossentropy + one-hot 레이블 사용
                     False이면 SparseCategoricalCrossentropy 사용

    Returns:
        Callable (waveforms, labels) → (loss, accuracy)
    """
    clf_vars = list(classifier.trainable_variables)
    all_update_vars = yamnet_vars_to_train + clf_vars  # 업데이트할 전체 변수 목록
    n_yamnet = len(yamnet_vars_to_train)
    num_classes = classifier.output_shape[-1]

    @tf.function
    def train_step(
        waveforms: tf.Tensor,  # (B, T) float32 — 배치 파형
        labels: tf.Tensor,     # (B,)   int32  — 배치 레이블
    ) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            # YAMNet 전진 pass: 파형 → 임베딩 (배치별로 map 실행)
            # yamnet(wav)는 (프레임별예측, 임베딩, log-mel) 반환
            # 임베딩을 시간 축으로 평균 → 1024차원 고정 벡터
            embeddings = tf.map_fn(
                lambda wav: tf.reduce_mean(yamnet(wav)[1], axis=0),
                waveforms,
                dtype=tf.float32,
            )  # shape: (B, 1024)

            logits = classifier(embeddings, training=True)  # shape: (B, num_classes)

            # 클래스 불균형 보정: 각 샘플에 해당 클래스 가중치 적용
            sample_weights = tf.gather(weight_vector, tf.cast(labels, tf.int32))

            if use_sigmoid:
                # one-hot 레이블로 변환 → BinaryCrossentropy
                labels_oh = tf.one_hot(tf.cast(labels, tf.int32), depth=num_classes)
                per_sample_loss = tf.keras.losses.binary_crossentropy(labels_oh, logits)
            else:
                per_sample_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    labels, logits
                )
            # 가중 평균 loss
            loss = tf.reduce_mean(per_sample_loss * sample_weights)

        # 원하는 변수에 대해서만 기울기 계산
        grads = tape.gradient(loss, all_update_vars)
        # YAMNet 변수 업데이트 (작은 학습률)
        yamnet_optimizer.apply_gradients(zip(grads[:n_yamnet], yamnet_vars_to_train))
        # 분류기 변수 업데이트 (상대적으로 큰 학습률)
        clf_optimizer.apply_gradients(zip(grads[n_yamnet:], clf_vars))

        # 정확도 계산 (argmax 예측 vs 실제 레이블)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
        return loss, acc

    return train_step


def _make_eval_batch_fn(use_sigmoid: bool = False) -> Any:
    """검증용 배치 평가 tf.function을 반환합니다 (기울기 없음).

    학습 스텝과 동일한 전진 pass이지만 tape 없이 실행하여 메모리 절약.
    """
    @tf.function
    def _eval_batch(
        yamnet: Any,
        classifier: tf.keras.Model,
        waveforms: tf.Tensor,
        labels: tf.Tensor,
        weight_vector: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        embeddings = tf.map_fn(
            lambda wav: tf.reduce_mean(yamnet(wav)[1], axis=0),
            waveforms,
            dtype=tf.float32,
        )
        logits = classifier(embeddings, training=False)  # training=False: Dropout 비활성
        sample_weights = tf.gather(weight_vector, tf.cast(labels, tf.int32))

        if use_sigmoid:
            num_classes = classifier.output_shape[-1]
            labels_oh = tf.one_hot(tf.cast(labels, tf.int32), depth=num_classes)
            per_sample_loss = tf.keras.losses.binary_crossentropy(labels_oh, logits)
        else:
            per_sample_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)

        loss = tf.reduce_mean(per_sample_loss * sample_weights)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
        return loss, acc

    return _eval_batch


# 하위 호환성을 위한 모듈 수준 별칭 (import 호환성 유지)
@tf.function
def _eval_batch(
    yamnet: Any,
    classifier: tf.keras.Model,
    waveforms: tf.Tensor,
    labels: tf.Tensor,
    weight_vector: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """배치 평가 함수 (기울기 없음) — softmax 경로."""
    embeddings = tf.map_fn(
        lambda wav: tf.reduce_mean(yamnet(wav)[1], axis=0),
        waveforms,
        dtype=tf.float32,
    )
    logits = classifier(embeddings, training=False)
    per_sample_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
    sample_weights = tf.gather(weight_vector, tf.cast(labels, tf.int32))
    loss = tf.reduce_mean(per_sample_loss * sample_weights)
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
    return loss, acc


# ---------------------------------------------------------------------------
# 테스트 세트 평가 (numpy, 그래프 없음)
# ---------------------------------------------------------------------------

def evaluate_test(
    yamnet: Any,
    classifier: tf.keras.Model,
    X_test_wav: np.ndarray,
    y_test: np.ndarray,
    label_map: dict[int, int],
    class_map: dict[int, str],
    batch_size: int,
    logger: logging.Logger,
) -> None:
    """파인튜닝된 모델로 테스트 세트 클래스별 성능을 계산합니다.

    YAMNet 전진 pass → 임베딩 → 분류기 예측의 전체 파이프라인을 실행합니다.
    배치 단위로 처리하여 메모리 부족 방지.
    """
    from sklearn.metrics import classification_report, confusion_matrix  # type: ignore

    n = len(X_test_wav)
    all_preds: list[int] = []

    for start in range(0, n, batch_size):
        batch = tf.constant(X_test_wav[start: start + batch_size])
        # 배치 파형 → 임베딩
        emb = tf.map_fn(
            lambda wav: tf.reduce_mean(yamnet(wav)[1], axis=0),
            batch,
            dtype=tf.float32,
        )
        logits = classifier(emb, training=False)
        preds = tf.argmax(logits, axis=1).numpy().tolist()
        all_preds.extend(preds)

    y_pred = np.array(all_preds)
    acc = float(np.mean(y_pred == y_test))
    logger.info("Fine-tuned test accuracy: %.4f", acc)

    # dense label → 클래스 이름 역매핑
    inv_map = {v: k for k, v in label_map.items()}
    dense_labels = sorted(inv_map.keys())
    target_names = [class_map.get(inv_map[l], str(inv_map[l])) for l in dense_labels]

    cm = confusion_matrix(y_test, y_pred, labels=dense_labels)
    report_str = classification_report(
        y_test, y_pred, labels=dense_labels, target_names=target_names
    )
    logger.info("Confusion matrix:\n%s", cm)
    logger.info("Classification report:\n%s", report_str)

    logger.info("=" * 60)
    logger.info("FINE-TUNE EVALUATION COMPLETE")
    logger.info("  Test accuracy : %.4f", acc)
    for cls_label, metrics in classification_report(
        y_test, y_pred, labels=dense_labels,
        target_names=target_names, output_dict=True
    ).items():
        if isinstance(metrics, dict):
            logger.info(
                "  %-15s  P=%.3f  R=%.3f  F1=%.3f  N=%d",
                cls_label,
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1-score", 0),
                int(metrics.get("support", 0)),
            )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# 메인 파인튜닝 파이프라인
# ---------------------------------------------------------------------------

def run_finetune(config_path: str) -> None:
    """YAMNet 파인튜닝 파이프라인 전체를 실행합니다.

    [ 전체 흐름 ]
    1. manifest.csv 로드 → 폴드 분할
    2. 파형 로드 (librosa, 16kHz, 4초 = 64000샘플)
    3. 클래스 가중치 계산
    4. YAMNet + 분류기 로드
    5. GradientTape로 YAMNet 56개 변수 발견, 마지막 60% 선택
    6. 이중 옵티마이저 설정
    7. 수동 학습 루프 (Early Stopping 내장)
    8. 최적 가중치 복원 (메모리 내 스냅샷)
    9. 분류기 .h5 저장 + YAMNet 변수 .npz 저장
    10. 테스트 세트 평가
    11. TFLite INT8 재출력
    """
    cfg = load_config(config_path)
    logger = setup_logging(cfg, __name__)

    project_root = Path(config_path).parent
    ds_cfg    = cfg["dataset"]
    train_cfg = cfg["training"]
    yamnet_cfg = cfg["yamnet"]
    audio_cfg  = cfg["audio"]
    export_cfg = cfg["export"]
    ft_cfg = cfg.get("finetune", {})

    # 재현성 시드 고정
    seed: int = train_cfg["seed"]
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ── 하이퍼파라미터 설정 ──
    sample_rate: int     = audio_cfg["sample_rate"]          # 16000
    target_samples: int  = int(sample_rate * 4.0)            # 64000 (4초)
    batch_size: int      = ft_cfg.get("batch_size", 16)
    epochs: int          = ft_cfg.get("epochs", 30)
    yamnet_lr: float     = ft_cfg.get("yamnet_lr", 1e-5)     # YAMNet 미세 조정 학습률
    clf_lr: float        = ft_cfg.get("clf_lr", 1e-4)        # 분류기 헤드 학습률
    unfreeze_frac: float = ft_cfg.get("unfreeze_fraction", 0.20)  # 해제 비율
    patience: int        = ft_cfg.get("early_stopping_patience", 5)
    positive_boost: float = train_cfg.get("class_weight_positive_boost", 1.0)

    # ── 경로 설정 ──
    processed_dir    = project_root / ds_cfg["processed_dir"]
    manifest_path    = processed_dir / "manifest.csv"
    model_dir        = project_root / export_cfg["model_dir"]
    clf_src_path     = model_dir / export_cfg["keras_filename"]       # 원본 분류기
    clf_dst_path     = model_dir / "custom_classifier_finetuned.h5"  # 파인튜닝 결과
    yamnet_ckpt_dir  = model_dir / "yamnet_finetuned"
    yamnet_ckpt_dir.mkdir(parents=True, exist_ok=True)

    val_folds: list[int]  = ds_cfg["val_folds"]    # [9]
    test_folds: list[int] = ds_cfg["test_folds"]   # [10]

    class_map = get_class_map(cfg)
    use_sigmoid: bool = cfg["classifier"].get("output_activation", "softmax") == "sigmoid"
    logger.info("Output activation: %s", "sigmoid" if use_sigmoid else "softmax")

    # ── 1. Manifest → 폴드 분할 ──
    df = pd.read_csv(str(manifest_path))
    label_map = build_label_map(df)
    num_classes = len(label_map)

    logger.info("Label map: %s  |  num_classes=%d", label_map, num_classes)

    train_df, val_df, test_df = fold_split(df, val_folds, test_folds, logger)

    # ── 1b. background 샘플 수 상한 적용 (RAM OOM 방지) ──
    # 11,000개 background를 모두 로드하면 X_train ≈ 2.86 GB (float32 × 64000).
    # tf.data.from_tensor_slices() 가 추가 복사본을 만들어 총 5.7 GB → OOM.
    # background를 cap_bg 개로 제한해 RAM ≤ 1.5 GB 수준으로 유지한다.
    bg_cap: int = ft_cfg.get("background_cap", 2000)
    path_col = train_df["path"].str.replace("\\", "/", regex=False)
    is_snr   = path_col.str.contains("processed/snr_", regex=False)
    clean_df = train_df[~is_snr]
    bg_mask  = clean_df["class"] == "background"
    n_bg     = bg_mask.sum()
    if n_bg > bg_cap:
        bg_idx    = clean_df[bg_mask].sample(n=bg_cap, random_state=seed).index
        other_idx = clean_df[~bg_mask].index
        train_df  = train_df.loc[bg_idx.union(other_idx)]
        logger.info(
            "background 상한 적용: %d → %d  (cap=%d, train 전체: %d)",
            n_bg, bg_cap, bg_cap, len(train_df),
        )

    # ── 2. 파형 로드 ──
    logger.info("Loading training waveforms …")
    X_train, y_train = load_waveforms(
        train_df, project_root, sample_rate, target_samples, label_map, logger
    )
    logger.info("Loading validation waveforms …")
    X_val, y_val = load_waveforms(
        val_df, project_root, sample_rate, target_samples, label_map, logger
    )
    logger.info("Loading test waveforms …")
    X_test, y_test = load_waveforms(
        test_df, project_root, sample_rate, target_samples, label_map, logger
    )

    train_ds = make_tf_dataset(X_train, y_train, batch_size, shuffle=True,  seed=seed)
    val_ds   = make_tf_dataset(X_val,   y_val,   batch_size, shuffle=False, seed=seed)

    # ── 3. 클래스 가중치 ──
    class_weights = get_class_weights(y_train, label_map, positive_boost)
    # 가중치를 인덱스 기반 텐서로 변환 (tf.gather 사용을 위해)
    weight_vector = tf.constant(
        [class_weights.get(i, 1.0) for i in range(num_classes)],
        dtype=tf.float32,
    )
    logger.info("Class weights: %s", class_weights)

    # ── 4. 모델 로드 ──
    try:
        import tensorflow_hub as hub  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pip install tensorflow-hub") from exc

    logger.info("Loading YAMNet from %s …", yamnet_cfg["tfhub_url"])
    yamnet = hub.load(yamnet_cfg["tfhub_url"])  # TF Hub에서 YAMNet 로드
    logger.info("YAMNet loaded.")

    if not clf_src_path.exists():
        raise FileNotFoundError(
            f"Pre-trained classifier not found: {clf_src_path}\n"
            "Run `python -m src.classifier` first."
        )
    logger.info("Loading pre-trained classifier from %s …", clf_src_path)
    # compile=False: 파인튜닝 시 새 옵티마이저를 사용하므로 기존 컴파일 설정 무시
    classifier = tf.keras.models.load_model(str(clf_src_path), compile=False)
    logger.info("Classifier loaded — %d trainable vars", len(classifier.trainable_variables))

    # ── 5. YAMNet 변수 발견 후 마지막 N% 선택 ──
    logger.info("Discovering YAMNet variables via GradientTape …")
    all_yamnet_vars = discover_yamnet_vars(yamnet, logger)
    yamnet_vars_to_train = select_yamnet_trainable_vars(all_yamnet_vars, unfreeze_frac, logger)

    # ── 6. 이중 옵티마이저 설정 ──
    yamnet_optimizer = tf.keras.optimizers.Adam(learning_rate=yamnet_lr)  # YAMNet 미세 조정
    clf_optimizer    = tf.keras.optimizers.Adam(learning_rate=clf_lr)     # 분류기 헤드

    eval_batch_fn = _make_eval_batch_fn(use_sigmoid=use_sigmoid)

    train_step_fn = make_train_step(
        yamnet, classifier,
        yamnet_vars_to_train,
        yamnet_optimizer, clf_optimizer,
        weight_vector,
        use_sigmoid=use_sigmoid,
    )

    # ── 7. 학습 루프 ──
    best_val_loss = float("inf")
    patience_counter = 0

    # 최적 가중치 메모리 스냅샷 (SavedModel 저장 없이 numpy 배열로 보관)
    # → 에폭마다 파일 I/O 없이 빠른 복원 가능
    best_yamnet_vals: list[np.ndarray] = [v.numpy() for v in yamnet_vars_to_train]
    best_clf_vals:    list[np.ndarray] = [v.numpy() for v in classifier.trainable_variables]

    logger.info(
        "Starting fine-tuning — epochs=%d  batch=%d  "
        "yamnet_lr=%.0e  clf_lr=%.0e  unfreeze=%.0f%%",
        epochs, batch_size, yamnet_lr, clf_lr, unfreeze_frac * 100,
    )

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss_acc = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
        val_loss_acc   = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()

        # 학습 배치 처리
        for waveforms, labels in train_ds:
            loss, acc = train_step_fn(waveforms, labels)
            train_loss_acc[0].update_state(loss)
            train_loss_acc[1].update_state(acc)

        # 검증 배치 처리 (기울기 없음)
        for waveforms, labels in val_ds:
            loss, acc = eval_batch_fn(yamnet, classifier, waveforms, labels, weight_vector)
            val_loss_acc[0].update_state(loss)
            val_loss_acc[1].update_state(acc)

        t_epoch = time.perf_counter() - t0
        tr_loss = float(train_loss_acc[0].result())
        tr_acc  = float(train_loss_acc[1].result())
        vl_loss = float(val_loss_acc[0].result())
        vl_acc  = float(val_loss_acc[1].result())

        logger.info(
            "Epoch %3d/%d — %.0fs  "
            "train loss=%.4f acc=%.4f  |  val loss=%.4f acc=%.4f",
            epoch, epochs, t_epoch, tr_loss, tr_acc, vl_loss, vl_acc,
        )

        # Early Stopping: val_loss 개선 시 스냅샷 저장, 아니면 patience 증가
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_counter = 0
            # 메모리에 최적 가중치 스냅샷 저장 (numpy 배열)
            best_yamnet_vals = [v.numpy() for v in yamnet_vars_to_train]
            best_clf_vals    = [v.numpy() for v in classifier.trainable_variables]
            logger.info("  val_loss improved → best weights snapshot saved.")
        else:
            patience_counter += 1
            logger.info(
                "  val_loss did not improve (patience %d/%d)",
                patience_counter, patience,
            )
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    # ── 8. 최적 가중치 복원 ──
    # 스냅샷에서 변수에 직접 assign → 마지막 에폭이 아닌 최적 에폭 복원
    for var, val in zip(yamnet_vars_to_train, best_yamnet_vals):
        var.assign(val)
    for var, val in zip(classifier.trainable_variables, best_clf_vals):
        var.assign(val)
    logger.info("Best weights restored (val_loss=%.4f).", best_val_loss)

    # ── 9. 아티팩트 저장 ──
    # 분류기: Keras .h5 형식
    classifier.save(str(clf_dst_path))
    logger.info("Fine-tuned classifier saved → %s", clf_dst_path)

    # YAMNet 변수: .npz 형식 (이름을 키로 저장)
    # realtime_infer.YAMNetEngine에서 동일한 발견 단계로 변수명 매칭하여 복원
    yamnet_npz_path = yamnet_ckpt_dir / "yamnet_vars.npz"
    yamnet_vars_dict = {
        # 변수명의 '/'와 ':'를 안전한 문자로 치환 (파일 키에 사용 불가)
        v.name.replace("/", "__").replace(":", "_"): v.numpy()
        for v in all_yamnet_vars   # 56개 전체 저장 (완전한 모델 상태 보존)
    }
    np.savez(str(yamnet_npz_path), **yamnet_vars_dict)
    logger.info("Fine-tuned YAMNet variables saved → %s", yamnet_npz_path)

    # ── 10. 테스트 평가 ──
    logger.info("Evaluating on test set …")
    evaluate_test(
        yamnet, classifier,
        X_test, y_test,
        label_map, class_map,
        batch_size=batch_size,
        logger=logger,
    )

    # ── 11. TFLite INT8 재출력 (Raspberry Pi 배포용) ──
    logger.info("Exporting fine-tuned classifier → TFLite INT8 …")
    _export_tflite(classifier, project_root, cfg, logger)


def _export_tflite(
    classifier: tf.keras.Model,
    project_root: Path,
    cfg: dict,
    logger: logging.Logger,
) -> None:
    """파인튜닝된 분류기를 TFLite INT8로 재출력합니다.

    [ INT8 양자화란? ]
    32비트 부동소수점 가중치를 8비트 정수로 변환 → 모델 크기 4배 감소
    Raspberry Pi 4B의 제한된 메모리와 연산 성능에서 실시간 추론 가능.

    [ 보정 데이터 ]
    INT8 양자화에는 대표 입력 샘플(보정 데이터)이 필요합니다.
    원본 임베딩(frozen YAMNet 생성)을 사용해도 괜찮습니다.
    분류기 TFLite는 1024차원 임베딩을 입력으로 받으므로,
    어떤 YAMNet이 보정 임베딩을 생성했든 양자화 스케일이 유효합니다.
    """
    try:
        from .model_export import convert_classifier_to_tflite  # type: ignore
    except Exception as exc:
        logger.warning("Could not import model_export: %s — skipping TFLite export.", exc)
        return

    export_cfg  = cfg["export"]
    yamnet_cfg  = cfg["yamnet"]
    model_dir   = project_root / export_cfg["model_dir"]
    tflite_path = model_dir / export_cfg["tflite_filename"]

    # INT8 보정용 임베딩 로드
    npz_path = project_root / yamnet_cfg["cache_dir"] / "embeddings.npz"
    if not npz_path.exists():
        logger.warning("embeddings.npz not found at %s — skipping TFLite export.", npz_path)
        return

    X_calib: np.ndarray = np.load(str(npz_path))["X"].astype(np.float32)
    logger.info("Calibration embeddings loaded: %s", X_calib.shape)

    try:
        tflite_bytes, quant_mode = convert_classifier_to_tflite(classifier, X_calib, logger)
        tflite_path.write_bytes(tflite_bytes)
        size_mb = len(tflite_bytes) / (1024 ** 2)
        logger.info(
            "TFLite (%s) saved → %s  (%.2f MB)",
            quant_mode, tflite_path, size_mb,
        )
    except Exception as exc:
        logger.warning("TFLite export failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ADAS Sound Detector – YAMNet Fine-tuning"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run_finetune(_parse_args().config)
