"""
manifest.csv 재생성 스크립트
기존 형식 그대로 유지:
  path, class, class_id, fold, snr_db, original_file
"""

import os, csv
from pathlib import Path

BASE    = Path('/mnt/c/Users/Daniel Park/Desktop/SoundPJ-rebuilt')
PROC    = BASE / 'data' / 'processed'
OUT     = PROC / 'manifest.csv'
BACKUP  = PROC / 'manifest_old.csv'

CLASS_ID = {'background': 0, 'car_horn': 1, 'siren': 2}

SNR_CONDITIONS = {
    'clean':     'clean',
    'snr_+10dB': '10',
    'snr_+5dB':  '5',
    'snr_+0dB':  '0',
    'snr_-5dB':  '-5',
}

# 기존 manifest 백업
if OUT.exists() and not BACKUP.exists():
    import shutil
    shutil.copy(OUT, BACKUP)
    print(f"백업 생성: {BACKUP.name}")

rows = []

for snr_folder, snr_val in SNR_CONDITIONS.items():
    snr_dir = PROC / snr_folder
    if not snr_dir.is_dir():
        continue

    for fold_dir in sorted(snr_dir.iterdir(), key=lambda x: int(x.name.replace('fold',''))):
        if not fold_dir.is_dir():
            continue
        fold_num = fold_dir.name.replace('fold', '')

        for cls_dir in sorted(fold_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name
            if cls_name not in CLASS_ID:
                continue
            cls_id = CLASS_ID[cls_name]

            for wav in sorted(cls_dir.glob('*.wav')):
                # path는 Windows 스타일 상대경로 (기존 형식 유지)
                rel = wav.relative_to(BASE)
                win_path = str(rel).replace('/', '\\')
                rows.append({
                    'path':          win_path,
                    'class':         cls_name,
                    'class_id':      cls_id,
                    'fold':          fold_num,
                    'snr_db':        snr_val,
                    'original_file': wav.name,
                })

# 기록
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['path','class','class_id','fold','snr_db','original_file'])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n manifest.csv 재생성 완료: {len(rows):,}행")

# 클래스별 통계
from collections import Counter
cls_cnt = Counter(r['class'] for r in rows)
snr_cnt = Counter(r['snr_db'] for r in rows)

print("\n클래스별 샘플 수:")
for cls, n in sorted(cls_cnt.items()):
    print(f"  {cls:12s}: {n:7,}")

print("\nSNR 조건별 샘플 수:")
for snr in ['clean','10','5','0','-5']:
    print(f"  {snr:6s}: {snr_cnt.get(snr, 0):7,}")
