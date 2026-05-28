import os

base = '/mnt/c/Users/Daniel Park/Desktop/SoundPJ-rebuilt/data'

def show_tree(path, prefix="", depth=0, max_depth=4, max_files=5):
    if depth > max_depth:
        return
    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        return
    dirs = [i for i in items if os.path.isdir(os.path.join(path, i))]
    files = [i for i in items if os.path.isfile(os.path.join(path, i))]
    for d in dirs:
        print(f"{prefix}[DIR] {d}/")
        show_tree(os.path.join(path, d), prefix + "  ", depth+1, max_depth, max_files)
    if files:
        shown = files[:max_files]
        for f in shown:
            sz = os.path.getsize(os.path.join(path, f))
            print(f"{prefix}{f}  ({sz//1024}KB)")
        if len(files) > max_files:
            print(f"{prefix}... and {len(files)-max_files} more files (total: {len(files)})")

print("=== data/ 전체 구조 ===")
show_tree(base, max_depth=3, max_files=3)

# straffic 폴더 상세
straffic = os.path.join(base, 'raw', 'straffic')
if os.path.isdir(straffic):
    print(f"\n=== straffic 상세 ===")
    show_tree(straffic, max_depth=4, max_files=5)

# noise 관련 파일 찾기
print("\n=== 'noise' 키워드 파일/폴더 탐색 ===")
for root, dirs, files in os.walk(base):
    # UrbanSound8K 건너뜀 (보호)
    dirs[:] = [d for d in dirs if d != 'UrbanSound8K']
    for name in dirs + files:
        if 'noise' in name.lower():
            full = os.path.join(root, name)
            print(f"  {full}")
