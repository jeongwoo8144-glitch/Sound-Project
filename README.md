# ADAS Sound Detector 🔊

청각 장애인 운전자를 위한 **차량 내 위험 소리 감지 시스템**

YAMNet 파인튜닝 기반 실시간 오디오 분류 모델로, 사이렌·경적·배경 소리를 구분해 시각적 알림을 제공합니다.

---

## 주요 기능

- **3-class 분류**: `siren` / `car_horn` / `background`
- **YAMNet 파인튜닝**: Block 9~14 (60%) 해제, Dual LR (yamnet_lr=5e-6, clf_lr=2e-5)
- **Triple Ensemble (v7)**: DNN + Random Forest + Siren Specialist
- **TFLite INT8 배포**: 0.29MB — 스마트폰 엣지 추론 가능
- **데이터 증강**: car_horn fold별 균형 맞춤 (527개 증강) + background 합성 (10,000개)

---

## 모델 성능

| 모델 | Accuracy | Siren Recall | Horn Recall |
|------|----------|-------------|-------------|
| Phase 1 (frozen) | 68.7% | - | - |
| Phase 4 (fine-tuned) | 72.4% | - | - |
| **Triple Ensemble v7** | **94.0%** | **-** | **-** |

---

## 프로젝트 구조

```
SoundPJ-rebuilt/
├── src/                        # 핵심 모듈
│   ├── classifier.py           # DNN 분류기
│   ├── data_pipeline.py        # 데이터 로딩 및 전처리
│   ├── embedding.py            # YAMNet 임베딩 추출
│   ├── finetune.py             # YAMNet 파인튜닝
│   ├── model_export.py         # TFLite 변환
│   ├── realtime_infer.py       # 실시간 추론
│   ├── siren_specialist.py     # 사이렌 전문 모델
│   ├── ensemble_eval.py        # 트리플 앙상블 평가
│   └── analysis/               # 레이어 분석 (Layer Probe 등)
│
├── scripts/                    # 실행 스크립트
│   ├── augment_horn.py         # car_horn 데이터 증강
│   ├── synthesize_background.py # background 합성 데이터 생성
│   ├── rebuild_manifest.py     # manifest.csv 재생성
│   ├── lr_comparison_experiment.py  # lr 탐색 실험
│   ├── train_rf_save.py        # RF 분류기 학습
│   └── triple_ensemble_eval.py # 앙상블 평가
│
├── reports/                    # 보고서 생성 스크립트 및 PDF
│   ├── generate_finetune_report.py
│   ├── horn_augmentation_report.pdf
│   ├── background_synthesis_report.pdf
│   └── pdf/                    # 최종 보고서 PDF
│
├── models/                     # 모델 파일
│   ├── adas_detector_int8_finetuned.tflite  # 배포용 (0.29MB)
│   └── ...
│
├── data/
│   └── processed/
│       └── manifest.csv        # 학습 데이터 경로 목록
│
├── results/                    # 실험 결과
├── config.yaml                 # 전체 설정
└── requirements.txt
```

---

## 설치 및 실행

### 환경 설정

```bash
# Python 3.12 권장
pip install -r requirements.txt
```

### 데이터 준비

1. [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) 다운로드 후 `data/raw/UrbanSound8K/` 위치
2. 전처리 실행:

```bash
python src/data_pipeline.py
```

3. 데이터 증강 (선택):

```bash
python scripts/augment_horn.py          # car_horn 균형 맞춤
python scripts/synthesize_background.py # background 합성 10,000개
python scripts/rebuild_manifest.py      # manifest 갱신
```

### 학습

```bash
python src/finetune.py
```

### TFLite 변환

```bash
python src/model_export.py
```

### 실시간 추론 테스트

```bash
python src/realtime_infer.py
```

---

## 데이터셋

- **UrbanSound8K**: 10-fold cross validation, fold9=val, fold10=test
- **3 클래스**: siren (929개), car_horn (956개, 증강 후), background (12,000개, 합성 후)
- **SNR 조건**: clean / +10dB / +5dB / 0dB / -5dB

> ⚠️ 오디오 파일(.wav)과 원본 데이터셋은 용량 문제로 GitHub에 포함되지 않습니다.

---

## 기술 스택

- **TensorFlow 2.21** + YAMNet (TF Hub)
- **librosa** — 오디오 처리 및 증강
- **scikit-learn** — Random Forest
- **TFLite** — 엣지 배포
- **ReportLab** — PDF 보고서 생성

---

## 라이선스

본 프로젝트는 학술/연구 목적으로 제작되었습니다.  
UrbanSound8K 데이터셋은 별도의 라이선스를 따릅니다.
