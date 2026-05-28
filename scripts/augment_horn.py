"""
car_horn 클래스를 fold별 siren 수와 맞추는 augmentation 스크립트.
clean 폴더에 augmented 파일 생성 후, SNR 버전도 함께 생성.

augmentation 기법:
  1. time_stretch  : rate 0.85 ~ 1.15
  2. pitch_shift   : -3 ~ +3 semitones
  3. gain          : -4 ~ +4 dB
  4. combo         : time_stretch + pitch_shift 동시 적용
"""

import os, random, numpy as np, librosa, soundfile as sf

random.seed(42)
np.random.seed(42)

SR = 22050
DURATION = 4.0          # UrbanSound8K 표준 4초
N_SAMPLES = int(SR * DURATION)

PROCESSED = '/mnt/c/Users/Daniel Park/Desktop/SoundPJ-rebuilt/data/processed'
SNR_TAGS   = ['snr_+10dB', 'snr_+5dB', 'snr_+0dB', 'snr_-5dB']
SNR_DB     = [10, 5, 0, -5]


# ── 오디오 유틸 ─────────────────────────────────────────────────
def load_fixed(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]
    return y.astype(np.float32)


def add_snr_noise(signal, snr_db):
    sig_power = np.mean(signal ** 2) + 1e-10
    noise = np.random.randn(len(signal)).astype(np.float32)
    noise_power = np.mean(noise ** 2) + 1e-10
    scale = np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10)))
    return np.clip(signal + scale * noise, -1.0, 1.0)


# ── augmentation 변환 ────────────────────────────────────────────
AUG_POOL = [
    ('ts085',  lambda y: librosa.effects.time_stretch(y, rate=0.85)),
    ('ts090',  lambda y: librosa.effects.time_stretch(y, rate=0.90)),
    ('ts110',  lambda y: librosa.effects.time_stretch(y, rate=1.10)),
    ('ts115',  lambda y: librosa.effects.time_stretch(y, rate=1.15)),
    ('ps_m3',  lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=-3)),
    ('ps_m2',  lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=-2)),
    ('ps_p2',  lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=+2)),
    ('ps_p3',  lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=+3)),
    ('gain_m4',lambda y: np.clip(y * 10**(-4/20), -1, 1)),
    ('gain_p4',lambda y: np.clip(y * 10**(+4/20), -1, 1)),
    ('combo1', lambda y: librosa.effects.pitch_shift(
                    librosa.effects.time_stretch(y, rate=0.90), sr=SR, n_steps=+2)),
    ('combo2', lambda y: librosa.effects.pitch_shift(
                    librosa.effects.time_stretch(y, rate=1.10), sr=SR, n_steps=-2)),
]


def apply_aug(y, tag):
    fn = dict(AUG_POOL)[tag]
    out = fn(y)
    if len(out) < N_SAMPLES:
        out = np.pad(out, (0, N_SAMPLES - len(out)))
    else:
        out = out[:N_SAMPLES]
    return out.astype(np.float32)


# ── fold별 생성 ──────────────────────────────────────────────────
clean_base = os.path.join(PROCESSED, 'clean')
folds = sorted(os.listdir(clean_base), key=lambda x: int(x.replace('fold', '')))

total_created = 0

for fold in folds:
    siren_dir = os.path.join(clean_base, fold, 'siren')
    horn_dir  = os.path.join(clean_base, fold, 'car_horn')

    n_siren = len([f for f in os.listdir(siren_dir) if f.endswith('.wav')])
    horn_files = [f for f in os.listdir(horn_dir) if f.endswith('.wav')]
    n_horn  = len(horn_files)
    need    = n_siren - n_horn

    if need <= 0:
        print(f"{fold}: horn({n_horn}) >= siren({n_siren}), skip")
        continue

    print(f"{fold}: horn={n_horn}, siren={n_siren}, generating {need} samples...")

    # 어떤 aug 태그를 쓸지 사이클 결정
    aug_tags = [tag for tag, _ in AUG_POOL]
    # 필요한 수만큼 (source, aug_tag) 쌍 결정 — source는 순환, aug는 다양하게
    assignments = []
    for i in range(need):
        src  = horn_files[i % len(horn_files)]
        tag  = aug_tags[i % len(aug_tags)]
        assignments.append((src, tag))

    for idx, (src_name, aug_tag) in enumerate(assignments):
        src_path = os.path.join(horn_dir, src_name)
        y = load_fixed(src_path)
        y_aug = apply_aug(y, aug_tag)

        stem = os.path.splitext(src_name)[0]
        new_name = f"aug_{stem}_{aug_tag}_{idx:03d}.wav"

        # clean 저장
        sf.write(os.path.join(horn_dir, new_name), y_aug, SR)

        # SNR 버전 저장
        for snr_tag, snr_db in zip(SNR_TAGS, SNR_DB):
            snr_horn_dir = os.path.join(PROCESSED, snr_tag, fold, 'car_horn')
            os.makedirs(snr_horn_dir, exist_ok=True)
            y_noisy = add_snr_noise(y_aug, snr_db)
            sf.write(os.path.join(snr_horn_dir, new_name), y_noisy, SR)

        total_created += 1

    new_count = len([f for f in os.listdir(horn_dir) if f.endswith('.wav')])
    print(f"  → horn: {n_horn} → {new_count}  (aug {need}개 완료)")

print(f"\n전체 augmented: {total_created}개 (clean 1 + SNR 4 = {total_created * 5}개 파일)")

# ── 최종 검증 ────────────────────────────────────────────────────
print("\n=== 최종 fold별 균형 확인 (clean) ===")
for fold in folds:
    siren_n = len(os.listdir(os.path.join(clean_base, fold, 'siren')))
    horn_n  = len(os.listdir(os.path.join(clean_base, fold, 'car_horn')))
    status  = "✓" if abs(horn_n - siren_n) <= 1 else f"△ diff={siren_n - horn_n}"
    print(f"  {fold}: siren={siren_n}, car_horn={horn_n}  {status}")
