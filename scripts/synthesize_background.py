"""
background + noise 합성으로 새 background 샘플 10,000개 생성.
원본 데이터는 절대 수정하지 않음.

합성 기법:
  1. SNR Mixing      : background + noise를 지정 SNR로 혼합
  2. Random Crop     : 긴 noise 파일에서 무작위 4초 구간 추출
  3. Gain Norm       : 출력 RMS를 -20dBFS로 정규화
  4. Random Gain     : ±4dB 무작위 볼륨 변동 (녹음 환경 다양성)
  5. Noise Category Mix : TCAR / engine / road 3가지 노이즈 카테고리 균형 혼합
"""

import os, random, numpy as np, librosa, soundfile as sf
from pathlib import Path

random.seed(7)
np.random.seed(7)

SR       = 22050
DURATION = 4.0
N_SAMP   = int(SR * DURATION)
TARGET   = 10_000
PER_FOLD = TARGET // 10

BASE_PROC = Path('/mnt/c/Users/Daniel Park/Desktop/SoundPJ-rebuilt/data/processed')
NOISE_DIR = Path('/mnt/c/Users/Daniel Park/Desktop/SoundPJ-rebuilt/data/noise')

# SNR 분포: 실도로 환경을 폭넓게 커버 (-5 ~ +15 dB)
SNR_CHOICES = [-5, -3, 0, 3, 5, 8, 10, 12, 15]
# 노이즈 카테고리 가중치 (TCAR:engine:road = 2:2:1)
NOISE_CATS  = ['TCAR', 'engine', 'road']
CAT_WEIGHTS = [2, 2, 1]


# ── 오디오 유틸 ─────────────────────────────────────────────────
def load_fixed(path, sr=SR):
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if len(y) < N_SAMP:
        y = np.pad(y, (0, N_SAMP - len(y)))
    else:
        start = random.randint(0, len(y) - N_SAMP) if len(y) > N_SAMP else 0
        y = y[start: start + N_SAMP]
    return y.astype(np.float32)


def rms_db(y):
    rms = np.sqrt(np.mean(y ** 2) + 1e-10)
    return 20 * np.log10(rms)


def normalize_rms(y, target_db=-20.0):
    current = rms_db(y)
    gain = 10 ** ((target_db - current) / 20)
    return np.clip(y * gain, -1.0, 1.0)


def mix_snr(signal, noise, snr_db):
    """signal + noise를 snr_db 비율로 혼합 후 RMS 정규화"""
    sig_p   = np.mean(signal ** 2) + 1e-10
    noi_p   = np.mean(noise  ** 2) + 1e-10
    scale   = np.sqrt(sig_p / (noi_p * 10 ** (snr_db / 10)))
    mixed   = signal + scale * noise
    return normalize_rms(mixed)


def random_gain(y, max_db=4.0):
    db  = random.uniform(-max_db, max_db)
    return np.clip(y * 10 ** (db / 20), -1.0, 1.0)


# ── 파일 목록 수집 ──────────────────────────────────────────────
bg_files = []
for fold_dir in sorted((BASE_PROC / 'clean').iterdir()):
    bg_dir = fold_dir / 'background'
    if bg_dir.is_dir():
        bg_files += list(bg_dir.glob('*.wav'))

noise_pools = {}
for cat in NOISE_CATS:
    cat_dir = NOISE_DIR / cat
    if cat_dir.is_dir():
        files = list(cat_dir.glob('*.wav'))
        noise_pools[cat] = files
        print(f"Noise [{cat}]: {len(files)} files")

print(f"Background source: {len(bg_files)} files")
print(f"Target synthesis: {TARGET} files ({PER_FOLD}/fold)")


# ── fold별 생성 ──────────────────────────────────────────────────
folds = sorted(
    [d for d in (BASE_PROC / 'clean').iterdir() if d.is_dir()],
    key=lambda x: int(x.name.replace('fold', ''))
)

total_created = 0
stats = {'snr_dist': {s: 0 for s in SNR_CHOICES}, 'cat_dist': {c: 0 for c in NOISE_CATS}}

for fold_dir in folds:
    out_dir = fold_dir / 'background'
    out_dir.mkdir(exist_ok=True)

    existing = len(list(out_dir.glob('*.wav')))
    print(f"\n{fold_dir.name}: existing={existing}, generating {PER_FOLD}...")

    for i in range(PER_FOLD):
        # 1. background 소스 무작위 선택 (해당 fold 파일 우선, 없으면 전체)
        fold_bg = [f for f in bg_files if fold_dir.name in str(f)]
        src_bg  = random.choice(fold_bg if fold_bg else bg_files)

        # 2. noise 카테고리 가중치 선택
        cat     = random.choices(NOISE_CATS, weights=CAT_WEIGHTS, k=1)[0]
        src_noi = random.choice(noise_pools[cat])
        stats['cat_dist'][cat] += 1

        # 3. SNR 무작위 선택
        snr = random.choice(SNR_CHOICES)
        stats['snr_dist'][snr] += 1

        # 4. 오디오 로드 (Random Crop 내장)
        y_bg  = load_fixed(src_bg)
        y_noi = load_fixed(src_noi)

        # 5. SNR Mixing
        y_mix = mix_snr(y_bg, y_noi, snr)

        # 6. Random Gain (±4dB)
        y_mix = random_gain(y_mix)

        # 7. 저장
        fname = f"synth_{fold_dir.name}_{cat}_{snr:+d}dB_{i:04d}.wav"
        sf.write(str(out_dir / fname), y_mix, SR)
        total_created += 1

    new_total = len(list(out_dir.glob('*.wav')))
    print(f"  → background: {existing} → {new_total}  (+{PER_FOLD}개)")


# ── 통계 출력 ────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"총 생성: {total_created}개")
print("\nSNR 분포:")
for snr, cnt in sorted(stats['snr_dist'].items()):
    bar = '█' * (cnt // 50)
    print(f"  {snr:+3d}dB: {cnt:5d}  {bar}")
print("\n노이즈 카테고리 분포:")
for cat, cnt in stats['cat_dist'].items():
    pct = cnt / total_created * 100
    print(f"  {cat:8s}: {cnt:5d}  ({pct:.1f}%)")

print("\n최종 background 파일 수:")
for fold_dir in folds:
    n = len(list((fold_dir / 'background').glob('*.wav')))
    print(f"  {fold_dir.name}: {n}")
