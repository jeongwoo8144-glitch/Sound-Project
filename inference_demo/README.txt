===================================================
  ADAS Sound Detector — Windows 실행 가이드
===================================================

[ 받은 파일 ]
  adas_detector.tflite  ← AI 모델 파일
  infer.py              ← 실행 스크립트
  README.txt            ← 이 파일


[ 1단계: Python 설치 확인 ]
  Python 3.9 이상 필요
  https://www.python.org/downloads/


[ 2단계: 패키지 설치 (최초 1회) ]
  명령 프롬프트(cmd) 또는 PowerShell을 열고:

  pip install tensorflow tensorflow-hub librosa sounddevice numpy


[ 3단계: 실행 ]

  # 오디오 파일 분류
  python infer.py --file 소리파일.wav

  # 마이크 실시간 감지
  python infer.py --mic

  # 감도 조정 (0.1=매우 민감, 0.7=둔감)
  python infer.py --mic --threshold 0.3


[ 출력 예시 ]

  파일 분류:
  ────────────────────────────────────────
    분류 결과
  ────────────────────────────────────────
    ⬜ 배경               2.1%  ▌
    📯 경적 (Car Horn)   94.3%  ████████████████████████████ ← 예측
    🚨 사이렌 (Siren)     3.6%  █
  ────────────────────────────────────────
  ⚠️  결과: 📯 경적 (Car Horn) 감지! (신뢰도: 94.3%)

  실시간:
  [17:30:15]  🚨 사이렌 (Siren) (87%)


[ 주의사항 ]
  - adas_detector.tflite 파일이 infer.py 와 같은 폴더에 있어야 합니다
  - 첫 실행 시 YAMNet 모델 다운로드로 1~2분 소요될 수 있습니다 (약 17MB)
  - 이후 실행부터는 캐시되어 빠르게 시작됩니다
  - 입력 오디오는 WAV, MP3, FLAC 모두 지원합니다
===================================================
