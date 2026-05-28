import os
from collections import defaultdict

base = '/mnt/c/Users/Daniel Park/Desktop/SoundPJ-rebuilt/data/processed'

# Check raw source data counts
raw_base = '/mnt/c/Users/Daniel Park/Desktop/SoundPJ-rebuilt/data/raw'
print("=== raw 데이터 폴더 구조 ===")
if os.path.isdir(raw_base):
    for d in sorted(os.listdir(raw_base)):
        print(f"  {d}")

print()
print("=== clean 조건 fold별 클래스 파일 수 ===")
clean_base = os.path.join(base, 'clean')
folds = sorted(os.listdir(clean_base), key=lambda x: int(x.replace('fold', '')))
for fold in folds:
    fp = os.path.join(clean_base, fold)
    if not os.path.isdir(fp):
        continue
    row = {}
    for cls in os.listdir(fp):
        cp = os.path.join(fp, cls)
        if os.path.isdir(cp):
            row[cls] = len([f for f in os.listdir(cp) if os.path.isfile(os.path.join(cp, f))])
    siren = row.get('siren', 0)
    horn = row.get('car_horn', 0)
    bg = row.get('background', 0)
    print(f"  {fold}: siren={siren}, car_horn={horn}, background={bg}")

print()
print("=== 전체 SNR 조건별 클래스 합계 ===")
for snr in sorted(os.listdir(base)):
    if snr == 'embeddings':
        continue
    snr_path = os.path.join(base, snr)
    if not os.path.isdir(snr_path):
        continue
    counts = defaultdict(int)
    for fold in os.listdir(snr_path):
        fp = os.path.join(snr_path, fold)
        if not os.path.isdir(fp):
            continue
        for cls in os.listdir(fp):
            cp = os.path.join(fp, cls)
            if os.path.isdir(cp):
                counts[cls] += len([f for f in os.listdir(cp) if os.path.isfile(os.path.join(cp, f))])
    siren = counts.get('siren', 0)
    horn = counts.get('car_horn', 0)
    bg = counts.get('background', 0)
    ratio = siren / horn if horn > 0 else 0
    print(f"  {snr}: siren={siren}, car_horn={horn}, bg={bg}  (siren/horn 비율: {ratio:.2f}x)")

# Check manifest
print()
manifest_path = '/mnt/c/Users/Daniel Park/Desktop/SoundPJ-rebuilt/data/processed/manifest.csv'
if os.path.isfile(manifest_path):
    import csv
    class_counts = defaultdict(int)
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_counts[row.get('label', row.get('class', ''))] += 1
    print("=== manifest.csv 클래스별 샘플 수 ===")
    for cls, n in sorted(class_counts.items()):
        print(f"  {cls}: {n}")
else:
    print("manifest.csv not found at expected path")
    # Search for it
    for root, dirs, files in os.walk('/mnt/c/Users/Daniel Park/Desktop/SoundPJ-rebuilt/data'):
        for f in files:
            if 'manifest' in f.lower():
                print(f"  Found: {os.path.join(root, f)}")
        dirs[:] = [d for d in dirs if d not in ['UrbanSound8K']]
