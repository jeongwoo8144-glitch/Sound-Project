"""Data pipeline for ADAS Sound Detector.

Step 1: UrbanSound8K에서 양성/음성 클래스 분리
  - 양성: car_horn, siren (target_classes)
  - 음성: 나머지 8개 → background 단일 클래스로 묶음

Step 2: Resample → pad/trim → SNR 노이즈 합성 → manifest.csv 출력

background 클래스 도입 이유:
  모델이 "아무 소리나 둘 중 하나로 분류"하는 것을 막기 위해
  "이건 아무것도 아닌 소리" 샘플을 학습시킴 → false positive 대폭 감소
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from .utils.config import get_class_map, load_config, setup_logging

TARGET_DURATION_SEC = 4.0
BACKGROUND_CLASS_ID = -1   # 내부 sentinel; manifest에는 class_id=99로 저장
BACKGROUND_LABEL = "background"
BACKGROUND_CLASS_ID_OUT = 99   # manifest 저장용 ID


# ---------------------------------------------------------------------------
# Step 1 – Metadata 분리
# ---------------------------------------------------------------------------

def split_metadata(
    metadata_csv: Path,
    target_class_ids: set[int],
    background_class_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """UrbanSound8K metadata를 양성/배경으로 분리.

    Args:
        metadata_csv: UrbanSound8K.csv 경로.
        target_class_ids: 양성 classID 집합 (car_horn, siren).
        background_class_ids: 배경으로 쓸 classID 집합.

    Returns:
        (target_df, background_df) 튜플.
    """
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    target_df = df[df["classID"].isin(target_class_ids)].copy().reset_index(drop=True)
    bg_df = df[df["classID"].isin(background_class_ids)].copy().reset_index(drop=True)
    return target_df, bg_df


def sample_background(
    bg_df: pd.DataFrame,
    cap: int,
    seed: int,
) -> pd.DataFrame:
    """background 클래스를 cap 개수로 균등 샘플링.

    fold 분포를 유지하기 위해 fold별로 비례 샘플링함.

    Args:
        bg_df: background 후보 DataFrame.
        cap: 최대 샘플 수.
        seed: 재현성을 위한 랜덤 시드.

    Returns:
        샘플링된 DataFrame.
    """
    if len(bg_df) <= cap:
        return bg_df

    # fold 비율 유지하며 샘플링 (pandas 3.x 호환 — groupby.apply 대신 명시적 루프)
    frac = cap / len(bg_df)
    chunks: list[pd.DataFrame] = []
    for _, group in bg_df.groupby("fold"):
        n = max(1, round(len(group) * frac))
        chunks.append(group.sample(min(n, len(group)), random_state=seed))

    sampled = pd.concat(chunks, ignore_index=True)
    # 혹시 초과하면 한 번 더 트림
    if len(sampled) > cap:
        sampled = sampled.sample(cap, random_state=seed).reset_index(drop=True)
    return sampled


# ---------------------------------------------------------------------------
# 오디오 처리 (기존과 동일)
# ---------------------------------------------------------------------------

def load_and_normalize(
    audio_path: Path,
    sample_rate: int,
    target_samples: int,
) -> Optional[np.ndarray]:
    """오디오 로드 → 리샘플 → pad/trim → float32 반환."""
    try:
        waveform, _ = librosa.load(
            str(audio_path), sr=sample_rate, mono=True, dtype=np.float32
        )
    except Exception:
        return None

    if len(waveform) >= target_samples:
        waveform = waveform[:target_samples]
    else:
        waveform = np.pad(waveform, (0, target_samples - len(waveform)), mode="constant")

    return waveform


def _load_noise_pool(
    noise_dir: Path,
    sample_rate: int,
    target_samples: int,
    logger,
) -> list[np.ndarray]:
    noise_files = sorted(
        set(noise_dir.glob("**/*.wav")) | set(noise_dir.glob("**/*.WAV"))
    )
    pool: list[np.ndarray] = []
    for p in noise_files:
        try:
            noise, _ = librosa.load(str(p), sr=sample_rate, mono=True, dtype=np.float32)
        except Exception as exc:
            logger.warning("Skipping noise file %s: %s", p.name, exc)
            continue
        if len(noise) < target_samples:
            noise = np.tile(noise, int(np.ceil(target_samples / len(noise))))
        pool.append(noise[:target_samples])
    logger.info("Loaded %d noise clips from %s", len(pool), noise_dir)
    return pool


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2) + 1e-9))


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """RMS 기반 SNR 합성 후 피크 정규화."""
    p_signal = _rms(signal) ** 2
    scale = np.sqrt(p_signal / (10 ** (snr_db / 10.0))) / _rms(noise)
    mixture = signal + scale * noise
    peak = np.max(np.abs(mixture))
    if peak > 1.0:
        mixture = mixture / peak
    return mixture.astype(np.float32)


def _sample_noise(
    pool: list[np.ndarray], rng: np.random.Generator
) -> Optional[np.ndarray]:
    return pool[int(rng.integers(0, len(pool)))] if pool else None


# ---------------------------------------------------------------------------
# 단일 파일 처리 (양성 / 배경 공통)
# ---------------------------------------------------------------------------

def _process_file(
    src_path: Path,
    fold: int,
    class_name: str,
    class_id_out: int,
    original_file: str,
    sample_rate: int,
    target_samples: int,
    processed_dir: Path,
    project_root: Path,
    noise_pool: list[np.ndarray],
    snr_levels: list[float],
    rng: np.random.Generator,
    logger,
    # background는 SNR 합성 생략 옵션 (데이터 크기 절약)
    augment: bool = True,
) -> list[dict]:
    """하나의 오디오 파일을 처리해 manifest 행 리스트를 반환."""
    waveform = load_and_normalize(src_path, sample_rate, target_samples)
    if waveform is None:
        logger.warning("Skipping (load failed): %s", src_path)
        return []

    rows: list[dict] = []

    # ── clean 저장 ──
    out_subdir = processed_dir / "clean" / f"fold{fold}" / class_name
    out_subdir.mkdir(parents=True, exist_ok=True)
    stem = Path(original_file).stem
    out_path = out_subdir / f"{stem}.wav"
    sf.write(str(out_path), waveform, sample_rate, subtype="PCM_16")

    rows.append({
        "path": str(out_path.relative_to(project_root)),
        "class": class_name,
        "class_id": class_id_out,
        "fold": fold,
        "snr_db": "clean",
        "original_file": original_file,
    })

    # ── SNR 합성 (background는 clean 1개만) ──
    if augment and noise_pool:
        for snr_db in snr_levels:
            noise_clip = _sample_noise(noise_pool, rng)
            mixed = mix_at_snr(waveform, noise_clip, snr_db)

            noisy_subdir = (
                processed_dir / f"snr_{int(snr_db):+d}dB" / f"fold{fold}" / class_name
            )
            noisy_subdir.mkdir(parents=True, exist_ok=True)
            noisy_path = noisy_subdir / f"{stem}.wav"
            sf.write(str(noisy_path), mixed, sample_rate, subtype="PCM_16")

            rows.append({
                "path": str(noisy_path.relative_to(project_root)),
                "class": class_name,
                "class_id": class_id_out,
                "fold": fold,
                "snr_db": snr_db,
                "original_file": original_file,
            })

    return rows


# ---------------------------------------------------------------------------
# STRAFFIC (DEMAND) — long-form WAV → 4-second clip segments
# ---------------------------------------------------------------------------

def _process_straffic(
    straffic_dir: Path,
    processed_dir: Path,
    project_root: Path,
    sample_rate: int,
    target_samples: int,
    num_folds: int,
    logger,
) -> list[dict]:
    """DEMAND STRAFFIC 파일을 4초 클립으로 분할해 background 샘플로 추가.

    Args:
        straffic_dir: STRAFFIC ch*.wav 파일들이 있는 디렉토리.
        processed_dir: 출력 processed 디렉토리.
        project_root: manifest 경로 기준점.
        sample_rate: 목표 샘플레이트 (16000 Hz).
        target_samples: 4초 = 64000 샘플.
        num_folds: fold 배정에 사용할 fold 수 (기본 10).
        logger: 로거.

    Returns:
        manifest 행 리스트.
    """
    wav_files = sorted(straffic_dir.glob("ch*.wav"))
    if not wav_files:
        logger.warning("STRAFFIC 파일 없음: %s", straffic_dir)
        return []

    rows: list[dict] = []
    fold_counter = 0  # round-robin fold 배정

    for wav_file in wav_files:
        try:
            audio, sr = librosa.load(str(wav_file), sr=sample_rate, mono=True, dtype=np.float32)
        except Exception as exc:
            logger.warning("STRAFFIC 로드 실패 %s: %s", wav_file.name, exc)
            continue

        n_clips = len(audio) // target_samples
        for i in range(n_clips):
            clip = audio[i * target_samples: (i + 1) * target_samples]

            fold = (fold_counter % num_folds) + 1
            fold_counter += 1

            out_subdir = processed_dir / "clean" / f"fold{fold}" / BACKGROUND_LABEL
            out_subdir.mkdir(parents=True, exist_ok=True)

            stem = f"{wav_file.stem}_clip{i:04d}"
            out_path = out_subdir / f"{stem}.wav"
            sf.write(str(out_path), clip, sample_rate, subtype="PCM_16")

            rows.append({
                "path": str(out_path.relative_to(project_root)),
                "class": BACKGROUND_LABEL,
                "class_id": BACKGROUND_CLASS_ID_OUT,
                "fold": fold,
                "snr_db": "clean",
                "original_file": f"{wav_file.name}_clip{i:04d}",
            })

    logger.info(
        "STRAFFIC 처리 완료: %d 파일 → %d 클립 (4초)",
        len(wav_files), len(rows),
    )
    return rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str) -> None:
    """전체 데이터 파이프라인 실행.

    클래스 구성:
      양성: target_classes (car_horn, siren) — SNR 합성 포함
      음성: background_classes → 'background' 단일 클래스 — clean only
    """
    cfg = load_config(config_path)
    logger = setup_logging(cfg, __name__)

    project_root = Path(config_path).parent
    ds_cfg = cfg["dataset"]
    audio_cfg = cfg["audio"]
    aug_cfg = cfg["augmentation"]

    sample_rate: int = audio_cfg["sample_rate"]
    target_samples = int(TARGET_DURATION_SEC * sample_rate)
    snr_levels: list[float] = aug_cfg["snr_levels_db"]
    seed: int = cfg["training"]["seed"]

    raw_dir = project_root / ds_cfg["raw_dir"]
    noise_dir = project_root / ds_cfg["noise_dir"]
    processed_dir = project_root / ds_cfg["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── 클래스 맵 ──
    class_map = get_class_map(cfg)                          # {id: name} 양성 클래스
    target_class_ids = set(class_map.keys())

    bg_cfg: dict = ds_cfg.get("background_classes", {})
    bg_class_ids: set[int] = {int(v) for v in bg_cfg.values()}
    bg_cap: int = ds_cfg.get("background_samples_cap", 800)

    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    logger.info("양성 클래스: %s", class_map)
    logger.info("배경 클래스 IDs: %s → '%s' (cap=%d)", bg_class_ids, BACKGROUND_LABEL, bg_cap)

    metadata_csv = raw_dir / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
    target_df, bg_df_full = split_metadata(metadata_csv, target_class_ids, bg_class_ids)
    bg_df = sample_background(bg_df_full, bg_cap, seed)

    logger.info(
        "데이터 분포  양성=%d  배경후보=%d  배경샘플=%d",
        len(target_df), len(bg_df_full), len(bg_df),
    )

    noise_pool = _load_noise_pool(noise_dir, sample_rate, target_samples, logger)
    if not noise_pool:
        logger.warning("노이즈 파일 없음 — SNR 합성 건너뜀.")

    audio_base = raw_dir / "UrbanSound8K" / "audio"
    manifest_rows: list[dict] = []
    skip_count = 0
    class_counts: dict[str, int] = {}

    # ── 양성 클래스 처리 (SNR 합성 포함) ──
    for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="양성 클래스 처리"):
        src_path = audio_base / f"fold{row['fold']}" / row["slice_file_name"]
        class_name = class_map[int(row["classID"])]
        rows = _process_file(
            src_path=src_path,
            fold=int(row["fold"]),
            class_name=class_name,
            class_id_out=int(row["classID"]),
            original_file=row["slice_file_name"],
            sample_rate=sample_rate,
            target_samples=target_samples,
            processed_dir=processed_dir,
            project_root=project_root,
            noise_pool=noise_pool,
            snr_levels=snr_levels,
            rng=rng,
            logger=logger,
            augment=True,   # 양성: SNR 합성 O
        )
        if not rows:
            skip_count += 1
        else:
            manifest_rows.extend(rows)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # ── 배경 클래스 처리 (clean only, SNR 합성 X) ──
    for _, row in tqdm(bg_df.iterrows(), total=len(bg_df), desc="배경 클래스 처리"):
        src_path = audio_base / f"fold{row['fold']}" / row["slice_file_name"]
        rows = _process_file(
            src_path=src_path,
            fold=int(row["fold"]),
            class_name=BACKGROUND_LABEL,
            class_id_out=BACKGROUND_CLASS_ID_OUT,
            original_file=row["slice_file_name"],
            sample_rate=sample_rate,
            target_samples=target_samples,
            processed_dir=processed_dir,
            project_root=project_root,
            noise_pool=noise_pool,
            snr_levels=snr_levels,
            rng=rng,
            logger=logger,
            augment=False,   # 배경: clean 1개만 (데이터 균형 유지)
        )
        if not rows:
            skip_count += 1
        else:
            manifest_rows.extend(rows)
            class_counts[BACKGROUND_LABEL] = class_counts.get(BACKGROUND_LABEL, 0) + 1

    # ── STRAFFIC (DEMAND 도로 교통 소음) background 추가 ──
    straffic_dir_key = ds_cfg.get("straffic_dir")
    if straffic_dir_key:
        straffic_dir = project_root / straffic_dir_key
        if straffic_dir.exists():
            straffic_rows = _process_straffic(
                straffic_dir=straffic_dir,
                processed_dir=processed_dir,
                project_root=project_root,
                sample_rate=sample_rate,
                target_samples=target_samples,
                num_folds=10,
                logger=logger,
            )
            manifest_rows.extend(straffic_rows)
            class_counts[BACKGROUND_LABEL] = class_counts.get(BACKGROUND_LABEL, 0) + len(straffic_rows)
        else:
            logger.warning("straffic_dir 경로 없음: %s", straffic_dir)

    # ── Manifest 저장 ──
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = processed_dir / "manifest.csv"
    manifest_df.to_csv(str(manifest_path), index=False)

    # ── 요약 ──
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Skipped       : %d files", skip_count)
    logger.info("  Manifest rows : %d", len(manifest_df))
    logger.info("  클래스별 원본 파일 수:")
    for cls, cnt in sorted(class_counts.items()):
        aug_note = f"(×{1 + len(snr_levels)} with SNR aug)" if cls != BACKGROUND_LABEL else "(clean only)"
        logger.info("    %-15s %4d원본  %s", cls, cnt, aug_note)
    logger.info("")
    logger.info("  최종 학습 샘플 수 (manifest 기준):")
    for cls, grp in manifest_df.groupby("class"):
        logger.info("    %-15s %d", cls, len(grp))
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADAS Sound Detector – Data Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(_parse_args().config)
