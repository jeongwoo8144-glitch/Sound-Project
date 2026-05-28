"""
ADAS Sound Detector — 전체 프로젝트 종합 보고서 PDF 생성기
시행착오, 에러 해결, 최종 결과 포함
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT

# ── 폰트 등록 ──
pdfmetrics.registerFont(TTFont("Mg",   "/mnt/c/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("MgB",  "/mnt/c/Windows/Fonts/malgunbd.ttf"))
pdfmetrics.registerFont(TTFont("MgSL", "/mnt/c/Windows/Fonts/malgunsl.ttf"))

OUTPUT = "/mnt/c/Users/Daniel Park/Downloads/SoundPJ-rebuilt/ADAS_Full_Report.pdf"

# ── 색상 팔레트 ──
C_NAVY    = colors.HexColor("#1a3a5c")
C_BLUE    = colors.HexColor("#2e6da4")
C_LBLUE   = colors.HexColor("#eaf2fb")
C_GREEN   = colors.HexColor("#1e8449")
C_LGREEN  = colors.HexColor("#d5f5e3")
C_RED     = colors.HexColor("#c0392b")
C_LRED    = colors.HexColor("#fdecea")
C_ORANGE  = colors.HexColor("#d35400")
C_LORANGE = colors.HexColor("#fef5e7")
C_GRAY    = colors.HexColor("#f4f6f8")
C_DGRAY   = colors.HexColor("#7f8c8d")
C_BORDER  = colors.HexColor("#bdc3c7")
C_YELLOW  = colors.HexColor("#fef9e7")

W_PAGE = A4[0] - 40*mm

def S(name, **kw):
    base = dict(fontName="Mg", fontSize=9.5, leading=15, textColor=colors.black,
                spaceAfter=4, alignment=TA_JUSTIFY)
    presets = {
        "title":    dict(fontName="MgB", fontSize=22, textColor=C_NAVY, spaceAfter=3,
                         alignment=TA_CENTER, leading=28),
        "subtitle": dict(fontName="Mg",  fontSize=12, textColor=C_BLUE, spaceAfter=2,
                         alignment=TA_CENTER, leading=18),
        "meta":     dict(fontName="Mg",  fontSize=8.5, textColor=C_DGRAY, spaceAfter=0,
                         alignment=TA_CENTER),
        "h1":       dict(fontName="MgB", fontSize=13, textColor=colors.white,
                         backColor=C_NAVY, spaceBefore=14, spaceAfter=7,
                         leading=20, leftIndent=-12, rightIndent=-12, borderPad=7),
        "h2":       dict(fontName="MgB", fontSize=11, textColor=C_BLUE,
                         spaceBefore=10, spaceAfter=4, leading=16, borderPad=2),
        "h3":       dict(fontName="MgB", fontSize=10, textColor=C_NAVY,
                         spaceBefore=6, spaceAfter=3, leading=14),
        "body":     dict(),
        "body_c":   dict(alignment=TA_CENTER),
        "code":     dict(fontName="Courier", fontSize=8.2, textColor=colors.HexColor("#1a252f"),
                         backColor=C_GRAY, leading=12.5, spaceAfter=5,
                         leftIndent=8, rightIndent=8, borderPad=6, alignment=TA_LEFT),
        "code_err": dict(fontName="Courier", fontSize=8.2, textColor=C_RED,
                         backColor=colors.HexColor("#fff5f5"), leading=12.5, spaceAfter=5,
                         leftIndent=8, rightIndent=8, borderPad=6, alignment=TA_LEFT),
        "code_ok":  dict(fontName="Courier", fontSize=8.2, textColor=C_GREEN,
                         backColor=colors.HexColor("#f0fff4"), leading=12.5, spaceAfter=5,
                         leftIndent=8, rightIndent=8, borderPad=6, alignment=TA_LEFT),
        "warn":     dict(fontName="Mg", fontSize=9.5, textColor=C_ORANGE, leading=15),
        "ok":       dict(fontName="Mg", fontSize=9.5, textColor=C_GREEN, leading=15),
        "err":      dict(fontName="Mg", fontSize=9.5, textColor=C_RED, leading=15),
        "bullet":   dict(fontName="Mg", fontSize=9.5, leading=15, spaceAfter=2,
                         leftIndent=14, bulletIndent=0, alignment=TA_LEFT),
        "caption":  dict(fontName="MgSL", fontSize=8, textColor=C_DGRAY,
                         alignment=TA_CENTER, spaceAfter=7),
        "toc":      dict(fontName="Mg", fontSize=10, leading=18, spaceAfter=1,
                         leftIndent=4),
        "toc2":     dict(fontName="Mg", fontSize=9.5, leading=16, spaceAfter=1,
                         leftIndent=16, textColor=C_BLUE),
        "note":     dict(fontName="MgSL", fontSize=9, textColor=C_DGRAY, leading=14,
                         leftIndent=8, rightIndent=8, spaceAfter=4),
    }
    cfg = {**base, **presets.get(name, {}), **kw}
    return ParagraphStyle(name, **cfg)


def tbl(data, col_w, hdr=True, hdr_bg=C_BLUE, row_alt=True, extra_cmds=None):
    t = Table(data, colWidths=col_w)
    cmds = [
        ("FONTNAME",      (0,0), (-1,-1), "Mg"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.8),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("GRID",          (0,0), (-1,-1), 0.35, C_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 7),
        ("RIGHTPADDING",  (0,0), (-1,-1), 7),
    ]
    if row_alt:
        cmds.append(("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_GRAY]))
    if hdr:
        cmds += [
            ("BACKGROUND",  (0,0), (-1,0), hdr_bg),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "MgB"),
            ("FONTSIZE",    (0,0), (-1,0), 9),
        ]
    if extra_cmds:
        cmds += extra_cmds
    t.setStyle(TableStyle(cmds))
    return t


def HR(thick=1, color=C_BORDER, space=4):
    return HRFlowable(width=W_PAGE, thickness=thick, color=color,
                      spaceBefore=space, spaceAfter=space)


def build():
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=18*mm,
        title="ADAS Sound Detector 종합 프로젝트 보고서",
        author="ADAS Team",
    )
    story = []

    # ═══════════════════════════════════════════════════════════════
    # 표지
    # ═══════════════════════════════════════════════════════════════
    story += [
        Spacer(1, 20*mm),
        Paragraph("ADAS Sound Detector", S("title")),
        Paragraph("종합 프로젝트 보고서", S("subtitle")),
        Spacer(1, 4*mm),
        HR(2, C_NAVY),
        Spacer(1, 4*mm),
        Paragraph(
            "청각 장애 운전자를 위한 실시간 경적·사이렌 감지 시스템<br/>"
            "YAMNet Fine-tuning + Custom DNN Classifier + TFLite INT8 배포",
            S("subtitle", fontSize=10, textColor=C_DGRAY)
        ),
        Spacer(1, 6*mm),
        Paragraph("작성일: 2026-05-21  |  최종 완료: Phase 4 YAMNet Finetune Round 2", S("meta")),
        Paragraph("학습 환경: Windows 11 + WSL2 Ubuntu 24.04 + NVIDIA RTX 3070 Laptop GPU", S("meta")),
        Spacer(1, 20*mm),
    ]

    # 목차
    story.append(Paragraph("목  차", S("h2", alignment=TA_CENTER, fontSize=13)))
    story.append(HR(0.5, C_BLUE))
    toc_items = [
        ("1", "프로젝트 개요"),
        ("2", "시스템 아키텍처"),
        ("3", "데이터셋 구성"),
        ("4", "학습 파이프라인 전체 흐름"),
        ("5", "Phase 0 — 초기 Classifier 학습 (CPU)"),
        ("6", "Phase 1 — YAMNet Finetune Round 1 (40% unfreeze)"),
        ("7", "Phase 2 — Embedding 재추출"),
        ("8", "Phase 3 — Classifier 재학습 (상세)"),
        ("9", "Phase 4 — YAMNet Finetune Round 2 (60% unfreeze)"),
        ("  9.1", "시행착오 전체 기록"),
        ("  9.2", "에러 상세 분석 및 해결"),
        ("  9.3", "최종 학습 결과"),
        ("10", "Siren Recall 심층 분석"),
        ("11", "핵심 설계 결정 사항"),
        ("12", "저장된 모델 파일 및 TFLite 배포"),
        ("13", "권장 후속 조치"),
        ("14", "전체 요약"),
    ]
    for num, title in toc_items:
        indent = 20 if num.startswith("  ") else 4
        story.append(
            Paragraph(f"{'&nbsp;'*4 if num.startswith('  ') else ''}{num}.  {title}",
                      S("toc2" if num.startswith("  ") else "toc", leftIndent=indent))
        )
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # 1. 프로젝트 개요
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("1.  프로젝트 개요", S("h1")))
    story.append(Paragraph(
        "본 시스템은 청각 장애를 가진 운전자가 운전 중 경적(car_horn) 및 긴급차량 사이렌(siren)을 "
        "인지하지 못해 발생할 수 있는 교통 사고를 예방하기 위해 개발되었습니다. "
        "차량 내 마이크를 통해 주변 음향을 실시간으로 수집하고, "
        "딥러닝 모델이 0.5초 간격으로 추론하여 해당 소리를 감지하면 시각적 경보를 발생시킵니다.",
        S("body")
    ))

    goal_data = [
        ["항목", "목표", "최종 결과", "달성"],
        ["검증 정확도", "98.5% 이상", "99.37% (Phase 4)", "✅"],
        ["Siren Recall (argmax)", "90% 이상", "68.7% (clean fold10)", "⚠️"],
        ["Car_horn Recall", "—", "97.0%", "✅"],
        ["TFLite 파일 크기", "5 MB 이하", "0.67 MB", "✅"],
        ["추론 latency", "200 ms 이하", "0.5s stride 설계", "✅"],
        ["배포 환경", "Raspberry Pi 4B", "INT8 양자화 완료", "✅"],
    ]
    story.append(tbl(goal_data, [W_PAGE*0.30, W_PAGE*0.22, W_PAGE*0.33, W_PAGE*0.15],
                     extra_cmds=[
                         ("BACKGROUND", (0,2), (-1,2), C_LGREEN),
                         ("BACKGROUND", (0,3), (-1,3), C_YELLOW),
                         ("BACKGROUND", (0,4), (-1,4), C_LGREEN),
                     ]))
    story.append(Paragraph("표 1-1. 프로젝트 목표 및 달성 현황", S("caption")))
    story.append(Paragraph(
        "⚠️  siren recall 수치는 4절에서 상세히 설명합니다. "
        "argmax 기반 68.7%와 실제 추론(threshold=0.20) 기반 recall은 다릅니다.",
        S("note")
    ))

    # ═══════════════════════════════════════════════════════════════
    # 2. 시스템 아키텍처
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("2.  시스템 아키텍처", S("h1")))
    story.append(Paragraph(
        "전체 시스템은 두 모듈로 구성됩니다. "
        "학습 파이프라인은 오프라인에서 YAMNet을 파인튜닝하고 TFLite로 변환합니다. "
        "실시간 추론 파이프라인은 마이크 입력을 슬라이딩 윈도우로 처리하여 경보를 발생시킵니다.",
        S("body")
    ))

    arch_data = [
        ["구성 요소", "역할", "세부 사항"],
        ["YAMNet (TF-Hub)", "오디오 임베딩 추출기", "MobileNet V1 기반, 56개 변수\n출력: (num_frames, 1024) → mean-pool → (1024,)"],
        ["Custom DNN Classifier", "3-class 분류기", "Dense(1024→512→256→64→3)\nBatchNorm, Dropout=0.4, Sigmoid 출력"],
        ["UrbanSound8K", "주 학습 데이터", "10개 fold, car_horn(429개) + siren(929개)\nbackground 8클래스 800개 캡"],
        ["STRAFFIC 소음", "배경 보강 데이터", "도로 교통 소음 16채널 × 75클립"],
        ["SNR 증강", "노이즈 견고성 강화", "clean + [+10, +5, 0, -5 dB] SNR 합성\n양성 클래스만 적용 (background는 clean only)"],
        ["TFLite INT8", "엣지 배포", "0.67 MB, Raspberry Pi 4B 대상"],
        ["Adaptive Threshold", "클래스별 독립 판정", "car_horn=0.45, siren=0.20\n(argmax 아닌 per-class threshold)"],
    ]
    story.append(tbl(arch_data, [W_PAGE*0.25, W_PAGE*0.27, W_PAGE*0.48]))
    story.append(Paragraph("표 2-1. 시스템 구성 요소", S("caption")))

    story.append(Paragraph("추론 흐름 (실시간)", S("h3")))
    story.append(Paragraph(
        "마이크 입력 (16 kHz 모노)  →  Ring Buffer (4.0s = 64,000 samples)  →<br/>"
        "슬라이딩 윈도우 (stride=0.5s)  →  YAMNet (파인튜닝)  →  mean-pool embedding (1024d)  →<br/>"
        "Classifier  →  3개 확률값 [car_horn, siren, background]  →<br/>"
        "per-class Adaptive Threshold  →  Majority Vote (3 frame debounce)  →  경보 발동",
        S("code")
    ))

    # ═══════════════════════════════════════════════════════════════
    # 3. 데이터셋 구성
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("3.  데이터셋 구성", S("h1")))

    story.append(Paragraph("3.1  UrbanSound8K 클래스 매핑", S("h2")))
    class_data = [
        ["클래스", "ClassID", "원본 수", "역할", "비고"],
        ["car_horn", "1", "429개", "양성 (Positive)", "경적 감지 대상"],
        ["siren", "8", "929개", "양성 (Positive)", "긴급차량 감지 대상"],
        ["air_conditioner", "0", "", "음성 → background", ""],
        ["children_playing", "2", "", "음성 → background", ""],
        ["dog_bark", "3", "", "음성 → background", ""],
        ["drilling", "4", "", "음성 → background", ""],
        ["engine_idling", "5", "", "음성 → background", ""],
        ["gun_shot", "6", "", "음성 → background", ""],
        ["jackhammer", "7", "", "음성 → background", ""],
        ["street_music", "9", "", "음성 → background", "800개 캡"],
    ]
    story.append(tbl(class_data, [W_PAGE*0.22, W_PAGE*0.12, W_PAGE*0.13, W_PAGE*0.26, W_PAGE*0.27],
                     extra_cmds=[
                         ("BACKGROUND", (0,1), (-1,1), C_LGREEN),
                         ("BACKGROUND", (0,2), (-1,2), C_LGREEN),
                     ]))
    story.append(Paragraph("표 3-1. UrbanSound8K 클래스 구성 (10개 fold, Fold 9=val, Fold 10=test)", S("caption")))

    story.append(Paragraph("3.2  학습/검증/테스트 분포", S("h2")))
    split_data = [
        ["클래스", "Train (Fold 1-8)", "Val (Fold 9)", "Test (Fold 10)", "합계"],
        ["car_horn", "1,820개 (aug)", "160개", "165개 (33 clean)", "~2,145"],
        ["siren", "3,820개 (aug)", "410개", "415개 (83 clean)", "~4,645"],
        ["background", "1,606개 (clean)", "196개", "198개", "2,000"],
        ["합계", "7,246개", "766개", "778개 / 314 clean", "—"],
    ]
    story.append(tbl(split_data, [W_PAGE*0.22, W_PAGE*0.25, W_PAGE*0.18, W_PAGE*0.25, W_PAGE*0.10]))
    story.append(Paragraph("표 3-2. 데이터 분포 (aug = SNR 증강 포함, clean = 원본만)", S("caption")))

    story.append(Paragraph(
        "SNR 증강 설명: 양성 클래스(car_horn, siren)에만 적용. "
        "STRAFFIC 도로 교통 소음을 [+10, +5, 0, -5 dB] 4가지 SNR 레벨로 합성. "
        "clean 원본 1개 → 총 5개(clean + 4 SNR). "
        "background는 이미 다양한 환경 소음이므로 clean만 사용.",
        S("note")
    ))

    # ═══════════════════════════════════════════════════════════════
    # 4. 학습 파이프라인 전체 흐름
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("4.  학습 파이프라인 전체 흐름", S("h1")))
    pipeline_data = [
        ["단계", "스크립트", "입력", "출력", "소요 시간"],
        ["Phase 0", "src/classifier.py\n(TF-Hub YAMNet)", "UrbanSound8K raw", "initial model\n(86.8% acc)", "~20분 (CPU)"],
        ["Phase 1", "src/finetune.py\n(40% unfreeze)", "Phase 0 모델\n+ raw waveforms", "yamnet_finetuned/\n(Round 1)", "~60분 (GPU)"],
        ["Phase 2", "src/embedding.py\n(finetuned YAMNet)", "finetuned YAMNet\n+ manifest.csv", "embeddings.npz\n(8,790 × 1024)", "~4분 (GPU)"],
        ["Phase 3", "src/classifier.py\n(DNN only)", "embeddings.npz", "custom_classifier.h5\n(87.5% acc)", "~2분 (GPU)"],
        ["Phase 4", "src/finetune.py\n(60% unfreeze)", "Phase 3 모델\n+ raw waveforms", "custom_classifier\n_finetuned.h5\nyamnet_vars.npz\nadas_detector.tflite", "~50분 (GPU)"],
    ]
    story.append(tbl(pipeline_data, [W_PAGE*0.10, W_PAGE*0.22, W_PAGE*0.20, W_PAGE*0.28, W_PAGE*0.20]))
    story.append(Paragraph("표 4-1. 전체 학습 파이프라인 (auto_train.sh 기준)", S("caption")))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # 5. Phase 0
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("5.  Phase 0 — 초기 Classifier 학습 (CPU)", S("h1")))
    story.append(Paragraph(
        "GPU 설정 전, TF-Hub의 원본 YAMNet으로 임베딩을 추출하고 "
        "Custom DNN을 학습한 베이스라인 모델입니다. "
        "이 단계의 목적은 데이터 파이프라인 검증과 초기 성능 기준점 확립입니다.",
        S("body")
    ))
    p0_data = [
        ["지표", "결과"],
        ["Test Accuracy", "86.8%"],
        ["Car_horn Recall", "92.1%"],
        ["Siren Recall", "76.1%"],
        ["특이사항", "Siren miss 대부분이 car_horn으로 분류됨\n→ 잘못된 클래스지만 어쨌든 경보는 울림"],
    ]
    story.append(tbl(p0_data, [W_PAGE*0.3, W_PAGE*0.7]))
    story.append(Paragraph("표 5-1. Phase 0 결과 (CPU, 원본 YAMNet 임베딩)", S("caption")))

    # ═══════════════════════════════════════════════════════════════
    # 6. Phase 1
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("6.  Phase 1 — YAMNet Finetune Round 1 (40% unfreeze)", S("h1")))
    story.append(Paragraph(
        "YAMNet 전체 56개 변수 중 하위 40%(layer 5~8 해당)를 해제하여 "
        "UrbanSound8K 데이터에 특화된 특징 추출기로 조정하는 단계입니다.",
        S("body")
    ))
    p1_cfg = [
        ["파라미터", "값"],
        ["Epochs", "50"],
        ["Unfreeze", "40% (layer 5~8, 23개 변수)"],
        ["YAMNet LR", "1e-5"],
        ["Classifier LR", "5e-5"],
        ["Early stopping patience", "8"],
    ]
    story.append(tbl(p1_cfg, [W_PAGE*0.4, W_PAGE*0.6]))
    story.append(Paragraph("표 6-1. Phase 1 설정", S("caption")))

    p1_epoch = [
        ["Epoch", "Val Acc", "Val Loss"],
        ["1",  "91.9%", "0.2437"],
        ["2",  "91.9%", "0.1133"],
        ["3",  "92.5%", "0.0845"],
        ["4",  "94.7%", "0.0617"],
        ["5",  "96.9%", "0.0343"],
        ["6",  "96.3%", "0.0417"],
        ["7",  "97.2%", "0.0193"],
        ["...", "...",   "..."],
        ["완료", "—", "Phase 2로 진행"],
    ]
    story.append(tbl(p1_epoch, [W_PAGE*0.2, W_PAGE*0.4, W_PAGE*0.4]))
    story.append(Paragraph("표 6-2. Phase 1 Epoch 진행 (초반부)", S("caption")))
    story.append(Paragraph(
        "결과물: models/yamnet_finetuned/ 에 Round 1 가중치 저장. "
        "Phase 2에서 이 가중치로 전체 학습 데이터의 임베딩을 재추출합니다.",
        S("note")
    ))

    # ═══════════════════════════════════════════════════════════════
    # 7. Phase 2
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("7.  Phase 2 — Embedding 재추출 (Finetuned YAMNet)", S("h1")))
    story.append(Paragraph(
        "Phase 1에서 파인튜닝된 YAMNet으로 전체 manifest.csv(8,790개 파일)의 "
        "임베딩을 다시 추출합니다. Phase 0에서 원본 YAMNet으로 추출한 임베딩보다 "
        "도메인 특화 특징을 담고 있어 Phase 3 분류기의 성능을 높입니다.",
        S("body")
    ))
    p2_data = [
        ["항목", "값"],
        ["처리 파일 수", "8,790개"],
        ["임베딩 차원", "(8,790 × 1,024) float32"],
        ["캐시 방식", "파일별 .npy 저장 → 재실행 시 캐시 히트"],
        ["저장 경로", "data/processed/embeddings/embeddings.npz"],
        ["소요 시간", "약 4분 (GPU, RTX 3070)"],
    ]
    story.append(tbl(p2_data, [W_PAGE*0.35, W_PAGE*0.65]))
    story.append(Paragraph("표 7-1. Phase 2 결과", S("caption")))

    # ═══════════════════════════════════════════════════════════════
    # 8. Phase 3
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("8.  Phase 3 — Classifier 재학습 (Finetuned Embeddings)", S("h1")))
    story.append(Paragraph(
        "Phase 2의 임베딩을 입력으로 Custom DNN만 학습합니다. "
        "YAMNet은 고정(frozen)된 상태이며, 분류기 헤드만 최적화합니다. "
        "에포크당 처리 시간이 매우 짧아(~1초) 빠른 반복 실험이 가능합니다.",
        S("body")
    ))

    story.append(Paragraph("8.1  학습 설정", S("h2")))
    p3_cfg = [
        ["파라미터", "값", "비고"],
        ["Hidden Units", "[512, 256, 64]", "3-layer DNN"],
        ["BatchNorm", "True", "각 Dense 층 후 적용"],
        ["Dropout", "0.4", "과적합 방지"],
        ["L2 Regularization", "1e-4", "가중치 정규화"],
        ["Loss Function", "Binary Crossentropy", "sigmoid 출력"],
        ["Learning Rate", "1e-3 → ReduceLR", "factor=0.5, patience=7"],
        ["Batch Size", "128", "embedding 학습"],
        ["Max Epochs", "200", "early stopping=15"],
        ["Class Weight Boost", "2.5×", "positive 클래스 페널티 강화"],
    ]
    story.append(tbl(p3_cfg, [W_PAGE*0.28, W_PAGE*0.28, W_PAGE*0.44]))
    story.append(Paragraph("표 8-1. Phase 3 분류기 설정", S("caption")))

    story.append(Paragraph("8.2  학습 진행 (주요 Epoch)", S("h2")))
    p3_epoch = [
        ["Epoch", "Train Acc", "Val Acc", "Val Loss", "비고"],
        ["1",  "82.96%", "90.86%", "0.3442", "Best"],
        ["2",  "88.69%", "92.04%", "0.2995", "Best"],
        ["6",  "91.61%", "92.65%", "0.2670", "Best"],
        ["10", "92.62%", "93.12%", "0.2549", "Best"],
        ["15", "93.06%", "93.17%", "0.2530", "Best"],
        ["19", "93.64%", "93.21%", "0.2484", "Best"],
        ["24", "94.23%", "93.52%", "0.2452", "Best"],
        ["26", "94.15%", "93.73%", "0.2389", "★ 최종 Best"],
        ["33", "—",      "—",      "—",      "ReduceLR 0.001→0.0005"],
        ["40", "—",      "—",      "—",      "ReduceLR 0.0005→0.00025"],
        ["41", "95.87%", "93.12%", "0.2551", "Early Stopping"],
    ]
    extra = [
        ("BACKGROUND", (0,8), (-1,8), C_LGREEN),
        ("FONTNAME",   (0,8), (-1,8), "MgB"),
    ]
    story.append(tbl(p3_epoch, [W_PAGE*0.12, W_PAGE*0.17, W_PAGE*0.17, W_PAGE*0.17, W_PAGE*0.37],
                     extra_cmds=extra))
    story.append(Paragraph("표 8-2. Phase 3 주요 Epoch 성능 (41 epochs, Early Stop)", S("caption")))

    story.append(Paragraph("8.3  Phase 3 최종 Test Set 결과", S("h2")))
    p3_test = [
        ["클래스", "Precision", "Recall", "F1-Score", "Support(N)"],
        ["car_horn", "0.581", "0.939", "0.718", "165"],
        ["siren",    "0.939", "0.773", "0.848", "415"],
        ["background", "0.923", "0.788", "0.850", "198"],
        ["전체 accuracy", "—", "0.812", "—", "778"],
        ["macro avg", "0.814", "0.834", "0.805", "778"],
        ["weighted avg", "0.859", "0.812", "0.821", "778"],
    ]
    extra3 = [
        ("BACKGROUND", (0,2), (-1,2), C_YELLOW),
        ("TEXTCOLOR",  (2,2), (2,2), C_GREEN),
        ("FONTNAME",   (2,2), (2,2), "MgB"),
    ]
    story.append(tbl(p3_test, [W_PAGE*0.25, W_PAGE*0.18, W_PAGE*0.18, W_PAGE*0.18, W_PAGE*0.21],
                     extra_cmds=extra3))
    story.append(Paragraph(
        "표 8-3. Phase 3 Test Set 결과 (augmented 778개, car_horn N=165, siren N=415, background N=198)",
        S("caption")
    ))
    story.append(Paragraph(
        "분석: car_horn precision이 0.581로 낮음 → siren을 car_horn으로 오분류하는 경향. "
        "siren recall 77.3%는 목표(90%)에 미달하나 class weight boost=2.5로 보완 예정. "
        "→ Phase 4에서 end-to-end 파인튜닝으로 개선 목표.",
        S("note")
    ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # 9. Phase 4
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("9.  Phase 4 — YAMNet Finetune Round 2 (60% unfreeze)", S("h1")))
    story.append(Paragraph(
        "Phase 4는 YAMNet의 상위 60%(layer 9~14, 33/56개 변수)를 해제하여 "
        "더 깊은 레이어까지 도메인 적응을 유도하는 단계입니다. "
        "이 단계에서 GPU 문제, venv 누락, 프로세스 강제 종료 등 다수의 시행착오가 발생했습니다.",
        S("body")
    ))

    story.append(Paragraph("9.1  시행착오 전체 기록", S("h2")))

    # 시도 타임라인 테이블
    attempt_data = [
        ["시도", "시작 시각", "설정", "결과/원인"],
        ["1차", "09:24", "batch=16, GPU 감지됨\nEpoch 1만 완료 (122s)", "사용자가 강제 종료\n(다른 설정 시도하려고)"],
        ["2차\n❌ GPU 없음", "09:30", "batch=8, LD_LIBRARY_PATH 미설정\n→ CPU 모드", "Epoch당 149초 (GPU의 2배)\nEpoch 6까지 진행 후 종료"],
        ["3차\n❌ venv 없음", "~09:47", "venv 미활성화 상태로 실행", "ModuleNotFoundError: numpy\n즉시 실패"],
        ["4차", "09:49", "batch=8, GPU 복구\n156s/epoch", "Epoch 1만 진행 후\n재설정을 위해 종료"],
        ["5차", "09:58", "batch=16, GPU\n152s→110s/epoch", "Epoch 16까지 진행\n사용자 '최고 가중치로 종료' 요청\n→ 디스크 미저장 (인메모리만)"],
        ["6차\n❌ venv 없음", "~10:28", "venv 미활성화 재시도", "ModuleNotFoundError: numpy\n즉시 실패"],
        ["7차\n✅ 최종", "10:38", "batch=16, GPU, venv ✅\n72s/epoch (안정화)", "Epoch 40 Early Stopping\n✅ 학습 완료 및 저장"],
    ]
    extra4 = [
        ("BACKGROUND", (0,2), (-1,2), C_LRED),
        ("BACKGROUND", (0,4), (-1,4), C_LRED),
        ("BACKGROUND", (0,5), (-1,5), C_LORANGE),
        ("BACKGROUND", (0,7), (-1,7), C_LGREEN),
        ("FONTNAME",   (0,7), (-1,7), "MgB"),
        ("ALIGN",      (0,0), (-1,-1), "LEFT"),
    ]
    story.append(tbl(attempt_data, [W_PAGE*0.12, W_PAGE*0.15, W_PAGE*0.35, W_PAGE*0.38],
                     extra_cmds=extra4))
    story.append(Paragraph("표 9-1. Phase 4 시도 이력 (총 7차, 2026-05-21)", S("caption")))

    story.append(Paragraph("9.2  에러 상세 분석 및 해결", S("h2")))

    # 에러 1
    story.append(KeepTogether([
        Paragraph("에러 1: GPU 라이브러리 로드 실패", S("h3")),
        Paragraph(
            "W0000  gpu_device.cc: Cannot dlopen some GPU libraries.\n"
            "Skipping registering GPU devices...",
            S("code_err")
        ),
        Paragraph(
            "원인: WSL2에서 nohup으로 프로세스 실행 시 LD_LIBRARY_PATH가 전달되지 않음. "
            "NVIDIA CUDA 라이브러리(.so 파일)들이 런타임에 로드되지 않아 TensorFlow가 GPU를 인식하지 못함.",
            S("body")
        ),
        Paragraph(
            "해결: 실행 전 LD_LIBRARY_PATH를 수동으로 export.\n"
            "export LD_LIBRARY_PATH=$(find ~/adas-env/lib -path '*/nvidia/*/lib' -type d | tr '\\n' ':'):/usr/lib/wsl/lib",
            S("code_ok")
        ),
        Paragraph(
            "→ RTX 3070 Laptop GPU (5,595 MB) 정상 인식, cuDNN 9.2.2 로드 확인.",
            S("ok")
        ),
    ]))

    # 에러 2
    story.append(KeepTogether([
        Paragraph("에러 2: ModuleNotFoundError (venv 미활성화)", S("h3")),
        Paragraph(
            "ModuleNotFoundError: No module named 'numpy'",
            S("code_err")
        ),
        Paragraph(
            "원인: Windows 터미널에서 직접 `python3 -m src.finetune` 실행 시 "
            "시스템 Python이 호출되어 프로젝트 의존성(numpy, tensorflow 등)을 찾지 못함. "
            "총 2회 발생 (2차, 6차 시도).",
            S("body")
        ),
        Paragraph(
            "해결: 항상 venv 활성화 후 실행.\n"
            "source ~/adas-env/bin/activate\npython3 -m src.finetune --config config.yaml",
            S("code_ok")
        ),
    ]))

    # 에러 3
    story.append(KeepTogether([
        Paragraph("에러 3: 강제 종료로 인한 최고 가중치 손실 (5차 시도)", S("h3")),
        Paragraph(
            "상황: 5차 시도에서 Epoch 16까지 val_acc=97.81%, val_loss=0.0282 달성.\n"
            "사용자가 '현재 최고 가중치로 종료' 요청 → pkill -f finetune 실행\n"
            "→ 가중치 저장 없이 프로세스 종료됨.",
            S("warn")
        ),
        Paragraph(
            "원인 분석: finetune.py의 가중치 저장 메커니즘\n"
            "# 훈련 중 인메모리 스냅샷 (디스크 저장 안 됨)\n"
            "if vl_loss < best_val_loss:\n"
            "    best_yamnet_vals = [v.numpy() for v in yamnet_vars_to_train]\n"
            "    best_clf_vals = [v.numpy() for v in classifier.trainable_variables]\n\n"
            "# 디스크 저장은 훈련 루프 종료 후에만 실행됨\n"
            "classifier.save(str(clf_dst_path))       # ← pkill 시 이 코드에 도달 못 함\n"
            "np.savez(str(yamnet_npz_path), ...)",
            S("code_err")
        ),
        Paragraph(
            "교훈: 훈련 중 강제 종료하면 인메모리 best weight가 모두 사라짐. "
            "Early stopping이 자연스럽게 완료될 때까지 기다려야 함.",
            S("note")
        ),
        Paragraph(
            "해결: 처음부터 다시 학습 시작. 사용자가 '이번엔 끝까지 가만 냅둘게' 동의.",
            S("ok")
        ),
    ]))

    story.append(Paragraph("9.3  최종 학습 결과 (7차 시도 — 성공)", S("h2")))
    cfg_final = [
        ["파라미터", "값"],
        ["Unfreeze", "60% (layer 9~14, 33/56 변수)"],
        ["YAMNet LR", "5×10⁻⁶"],
        ["Classifier LR", "2×10⁻⁵"],
        ["Batch Size", "16"],
        ["Max Epochs", "100"],
        ["Early Stopping patience", "10"],
        ["평균 Epoch 시간", "72초 (GPU 안정화 후)"],
        ["총 학습 시간", "약 50분 (10:38 ~ 11:48)"],
    ]
    story.append(tbl(cfg_final, [W_PAGE*0.4, W_PAGE*0.6]))
    story.append(Paragraph("표 9-2. Phase 4 최종 시도 설정", S("caption")))

    story.append(Paragraph("Phase 4 전체 Epoch 상세", S("h3")))
    p4_full = [
        ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Patience", "비고"],
        ["1",  "0.4220", "82.13%", "0.3086", "89.06%", "0/10", "Best"],
        ["2",  "0.2827", "87.28%", "0.2375", "91.87%", "0/10", "Best"],
        ["3",  "0.2432", "89.61%", "0.1187", "94.06%", "0/10", "Best"],
        ["4",  "0.2406", "89.83%", "0.1485", "94.69%", "1/10", ""],
        ["5",  "0.1923", "91.25%", "0.0636", "97.19%", "0/10", "Best"],
        ["6",  "0.1716", "92.06%", "0.0969", "96.25%", "1/10", ""],
        ["7",  "0.1832", "92.51%", "0.0606", "97.50%", "0/10", "Best"],
        ["8",  "0.1439", "92.80%", "0.0809", "96.56%", "1/10", ""],
        ["9",  "0.2129", "91.48%", "0.0709", "95.63%", "2/10", ""],
        ["10", "0.1217", "93.06%", "0.0462", "96.25%", "0/10", "Best"],
        ["11", "0.0743", "95.25%", "0.0472", "97.19%", "1/10", ""],
        ["12", "0.1179", "94.32%", "0.0516", "97.50%", "2/10", ""],
        ["13", "0.0824", "94.46%", "0.0439", "96.88%", "3/10", ""],
        ["14", "0.0883", "95.21%", "0.0377", "97.19%", "4/10", ""],
        ["15", "0.0880", "95.54%", "0.0311", "97.81%", "0/10", "Best"],
        ["16", "0.0611", "95.68%", "0.0231", "97.50%", "0/10", "Best"],
        ["17", "0.0575", "96.35%", "0.0503", "97.19%", "1/10", ""],
        ["18", "0.0450", "97.26%", "0.0371", "97.50%", "2/10", ""],
        ["19", "0.0464", "97.37%", "0.0233", "98.44%", "0/10", "Best"],
        ["20", "0.0459", "96.63%", "0.0222", "98.44%", "0/10", "Best"],
        ["21", "0.0540", "97.55%", "0.0187", "97.50%", "0/10", "Best"],
        ["22", "0.0566", "97.26%", "0.0179", "98.44%", "0/10", "Best"],
        ["23", "0.0453", "97.77%", "0.0256", "97.81%", "1/10", ""],
        ["24", "0.0431", "97.36%", "0.0169", "99.06%", "0/10", "Best"],
        ["25", "0.0280", "98.28%", "0.0133", "99.06%", "0/10", "Best"],
        ["26", "0.0154", "98.65%", "0.0148", "98.75%", "1/10", ""],
        ["27", "0.0332", "98.46%", "0.0250", "98.44%", "2/10", ""],
        ["28", "0.0170", "98.65%", "0.0145", "99.06%", "3/10", ""],
        ["29", "0.0214", "98.90%", "0.0130", "99.06%", "0/10", "Best"],
        ["30", "0.0159", "98.83%", "0.0065", "99.37%", "0/10", "★ 최종 Best"],
        ["31", "0.0314", "98.64%", "0.0748", "98.44%", "1/10", "spike"],
        ["32", "0.0199", "98.98%", "0.0182", "99.06%", "2/10", ""],
        ["33", "0.0176", "98.90%", "0.0188", "99.37%", "3/10", ""],
        ["34", "0.0217", "98.75%", "0.0326", "97.81%", "4/10", ""],
        ["35", "0.0138", "99.31%", "0.0131", "99.37%", "5/10", ""],
        ["36", "0.0261", "99.20%", "0.0322", "98.44%", "6/10", ""],
        ["37", "0.0335", "98.65%", "0.0346", "98.44%", "7/10", ""],
        ["38", "0.0192", "98.68%", "0.0267", "98.75%", "8/10", ""],
        ["39", "0.0260", "98.68%", "0.0291", "98.75%", "9/10", ""],
        ["40", "0.0194", "99.20%", "0.0122", "99.37%", "10/10", "Early Stop"],
    ]
    extra_p4 = [
        ("BACKGROUND", (0,30), (-1,30), C_LGREEN),
        ("FONTNAME",   (0,30), (-1,30), "MgB"),
        ("TEXTCOLOR",  (0,30), (-1,30), C_GREEN),
        ("BACKGROUND", (0,40), (-1,40), C_LRED),
        ("FONTNAME",   (0,40), (-1,40), "MgB"),
        ("FONTSIZE",   (0,0),  (-1,-1), 7.8),
        ("TOPPADDING",    (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]
    story.append(tbl(p4_full,
                     [W_PAGE*0.09, W_PAGE*0.12, W_PAGE*0.12, W_PAGE*0.12, W_PAGE*0.12, W_PAGE*0.12, W_PAGE*0.31],
                     extra_cmds=extra_p4))
    story.append(Paragraph(
        "표 9-3. Phase 4 전체 40 Epoch 성능 기록 (Best @ Epoch 30: val_loss=0.0065, val_acc=99.37%)",
        S("caption")
    ))

    story.append(PageBreak())

    # Phase 4 최종 테스트 결과
    story.append(Paragraph("Phase 4 Test Set 결과 (Fold 10, argmax 기준)", S("h3")))
    p4_test = [
        ["클래스", "Precision", "Recall", "F1-Score", "Support(N)", "비고"],
        ["car_horn",   "0.889", "0.970", "0.928", "33",  "✅ 목표 초과"],
        ["siren",      "0.905", "0.687", "0.781", "83",  "⚠️ argmax 기준\n(threshold 방식 ≠)"],
        ["background", "0.893", "0.970", "0.930", "198", "✅ FP 방지 우수"],
        ["전체 accuracy", "—", "0.895",  "—",    "314", ""],
        ["macro avg",  "0.896", "0.875", "0.879", "314", ""],
        ["weighted avg","0.896","0.895", "0.890", "314", ""],
    ]
    extra_t4 = [
        ("BACKGROUND", (0,2), (-1,2), C_LGREEN),
        ("BACKGROUND", (0,3), (-1,3), C_YELLOW),
        ("BACKGROUND", (0,4), (-1,4), C_LGREEN),
        ("TEXTCOLOR",  (2,3), (2,3), C_RED),
        ("FONTNAME",   (2,3), (2,3), "MgB"),
        ("ALIGN",      (0,0), (-1,-1), "LEFT"),
    ]
    story.append(tbl(p4_test,
                     [W_PAGE*0.18, W_PAGE*0.13, W_PAGE*0.13, W_PAGE*0.13, W_PAGE*0.13, W_PAGE*0.30],
                     extra_cmds=extra_t4))
    story.append(Paragraph(
        "표 9-4. Phase 4 최종 Test 결과 (clean 원음 314개, Fold 10)",
        S("caption")
    ))

    # 혼동 행렬
    story.append(Paragraph("Phase 4 혼동 행렬 (Confusion Matrix)", S("h3")))
    cm_data = [
        ["실제 \\ 예측", "car_horn", "siren", "background"],
        ["car_horn (33)", "32 ✅", "1", "0"],
        ["siren (83)", "3", "57 ✅", "23 ⚠️"],
        ["background (198)", "1", "5", "192 ✅"],
    ]
    extra_cm = [
        ("BACKGROUND", (1,1), (1,1), C_LGREEN),
        ("BACKGROUND", (2,2), (2,2), C_LGREEN),
        ("BACKGROUND", (3,3), (3,3), C_LGREEN),
        ("BACKGROUND", (3,2), (3,2), C_LRED),
        ("FONTNAME",   (3,2), (3,2), "MgB"),
        ("TEXTCOLOR",  (3,2), (3,2), C_RED),
    ]
    story.append(tbl(cm_data,
                     [W_PAGE*0.28, W_PAGE*0.24, W_PAGE*0.24, W_PAGE*0.24],
                     extra_cmds=extra_cm))
    story.append(Paragraph(
        "표 9-5. 혼동 행렬 — siren 83개 중 23개가 background로 분류됨 (miss의 주 원인)",
        S("caption")
    ))

    # ═══════════════════════════════════════════════════════════════
    # 10. Siren Recall 심층 분석
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("10.  Siren Recall 심층 분석", S("h1")))
    story.append(Paragraph(
        "이 섹션은 siren recall에 대해 가장 자주 나온 질문과 혼란을 정리합니다. "
        "68.7%라는 수치가 왜 목표(90%)에 미달하는지, "
        "실제 사용 환경에서 얼마나 될지 단계별로 설명합니다.",
        S("body")
    ))

    story.append(Paragraph("10.1  두 가지 측정값이 존재하는 이유", S("h2")))
    compare_data = [
        ["구분", "Val Acc 99.37%", "Test Siren Recall 68.7%"],
        ["무엇인가", "Phase 4 훈련 중 모니터링 지표", "최종 테스트셋 평가"],
        ["데이터", "Fold 9 (val set)", "Fold 10 (test set, clean)"],
        ["siren 샘플 수", "410개 (aug 포함)", "83개 (clean만)"],
        ["판정 방식", "argmax (3클래스 중 최고)", "argmax (동일)"],
        ["의미", "훈련 잘 됐다는 신호", "최종 성능 (보수적 평가)"],
    ]
    story.append(tbl(compare_data, [W_PAGE*0.22, W_PAGE*0.39, W_PAGE*0.39]))
    story.append(Paragraph("표 10-1. 두 지표의 차이", S("caption")))

    story.append(Paragraph("10.2  Argmax vs Threshold — 근본적 차이", S("h2")))
    story.append(Paragraph(
        "테스트 평가: argmax 방식 (3개 확률 중 가장 큰 클래스를 예측)",
        S("h3")
    ))
    story.append(Paragraph(
        "예시: car_horn=0.05,  siren=0.18,  background=0.77\n"
        "argmax = background  →  사이렌임에도 오분류 (MISS로 기록됨)",
        S("code_err")
    ))
    story.append(Paragraph(
        "실제 추론: per-class threshold 방식 (각 클래스 독립 판정)",
        S("h3")
    ))
    story.append(Paragraph(
        "예시 1: car_horn=0.05,  siren=0.18,  background=0.77\n"
        "  → siren(0.18) >= threshold(0.20)?  → NO   (경보 없음)\n\n"
        "예시 2: car_horn=0.05,  siren=0.22,  background=0.73\n"
        "  → siren(0.22) >= threshold(0.20)?  → YES  → 🚨 사이렌 경보!\n"
        "     (background가 더 높아도 siren이 0.20 넘으면 경보 발동)",
        S("code_ok")
    ))
    story.append(Paragraph(
        "결론: argmax에서 background(0.73)가 이겨 miss로 기록된 샘플도, "
        "threshold 방식에서는 siren이 0.20만 넘으면 올바르게 감지됩니다. "
        "따라서 실제 recall은 68.7%보다 높습니다.",
        S("body")
    ))

    story.append(Paragraph("10.3  Phase별 Siren Recall 비교", S("h2")))
    phase_recall = [
        ["Phase", "Siren Recall", "평가 방식", "Siren N", "비고"],
        ["Phase 3\n(Classifier)", "77.3%", "argmax", "415\n(aug 포함)", "더 현실적 (다양한 노이즈 포함)"],
        ["Phase 4\n(Finetune)", "68.7%", "argmax", "83\n(clean only)", "보수적 (노이즈 없는 원음만)"],
        ["실시간 추론\n(threshold=0.20)", "측정 필요", "per-class\nthreshold", "—", "실질적으로 더 높을 것"],
    ]
    extra_pr = [("BACKGROUND", (0,3), (-1,3), C_LBLUE)]
    story.append(tbl(phase_recall,
                     [W_PAGE*0.18, W_PAGE*0.17, W_PAGE*0.17, W_PAGE*0.13, W_PAGE*0.35],
                     extra_cmds=extra_pr))
    story.append(Paragraph("표 10-2. Phase별 Siren Recall 비교", S("caption")))
    story.append(Paragraph(
        "Phase 4 test recall(68.7%)이 Phase 3(77.3%)보다 낮은 이유: "
        "Phase 4는 clean 83개, Phase 3는 augmented 415개로 평가. "
        "샘플 수와 구성이 달라 단순 비교 불가.",
        S("note")
    ))

    story.append(Paragraph("10.4  Threshold 조정 가이드", S("h2")))
    thresh_guide = [
        ["Siren Threshold", "예상 효과", "권장 상황"],
        ["0.25 이상", "Precision↑, Recall↓\nFalse Positive 감소", "오경보가 너무 잦을 때"],
        ["0.20 (현재)", "Recall/Precision 균형", "기본 설정 (권장)"],
        ["0.15", "Recall↑, False Positive 증가 가능", "도로 위 긴급차량 민감 감지"],
        ["0.10", "매우 민감\n거의 모든 사이렌 감지", "극도의 안전 우선 상황"],
    ]
    extra_tg = [("BACKGROUND", (0,2), (-1,2), C_LBLUE), ("FONTNAME", (0,2), (-1,2), "MgB")]
    story.append(tbl(thresh_guide, [W_PAGE*0.22, W_PAGE*0.40, W_PAGE*0.38], extra_cmds=extra_tg))
    story.append(Paragraph("표 10-3. Siren Threshold 조정 가이드 (config.yaml 수정)", S("caption")))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # 11. 핵심 설계 결정 사항
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("11.  핵심 설계 결정 사항", S("h1")))

    decisions = [
        ("레이어 선택: 왜 60% unfreeze인가?",
         "YAMNet 56개 변수를 알파벳 순으로 정렬 후 하위 40%(23개, layer 1~8)는 frozen, "
         "상위 60%(33개, layer 9~14)는 trainable로 설정. "
         "일반적인 전이학습 원칙: 초기 레이어(edge, texture 등 저수준 특징)는 범용적이어서 재학습 불필요. "
         "후기 레이어(고수준 패턴)는 도메인 특화이므로 파인튜닝 효과가 큼. "
         "Round 1에서 40%로 먼저 적응시킨 뒤, Round 2에서 60%로 확장하여 더 깊은 적응 유도."),
        ("두 개의 Learning Rate를 쓰는 이유",
         "yamnet_lr=5e-6(매우 낮음) + clf_lr=2e-5(상대적으로 높음). "
         "YAMNet은 이미 대규모 오디오 데이터로 학습된 모델 → 큰 LR로 업데이트하면 기존 지식 파괴(catastrophic forgetting). "
         "Classifier는 처음부터 학습되는 것처럼 빠르게 수렴해야 함. "
         "두 최적화기를 분리함으로써 각 모듈에 최적화된 학습 속도 적용."),
        ("Background 클래스를 하나로 묶은 이유",
         "UrbanSound8K의 8개 비경보 클래스(air_conditioner, dog_bark 등)를 'background'로 통합. "
         "모델이 '경보 소리가 아닌 것'을 하나의 범주로 학습해 false positive를 줄이는 효과. "
         "800개 캡을 적용해 양성 클래스와 균형 유지."),
        ("SNR 증강을 양성 클래스에만 적용한 이유",
         "car_horn/siren은 실제 도로에서 항상 배경 소음과 함께 들림 → 노이즈 견고성 필수. "
         "background는 이미 다양한 환경에서 녹음된 소음 자체 → 추가 증강 불필요. "
         "clean + [+10, +5, 0, -5 dB] 4가지 SNR로 각 양성 샘플을 5배 증강."),
        ("Adaptive Threshold + Majority Voting",
         "argmax 대신 per-class 독립 threshold 사용 → siren과 car_horn을 동시에 감지 가능. "
         "debounce_frames=3 majority voting → 단순 노이즈 spike로 인한 오경보 방지. "
         "siren=0.20으로 낮게 설정 → recall 우선(놓치면 위험), car_horn=0.45로 높게 → precision 우선."),
        ("In-memory Weight Snapshot 방식의 한계",
         "finetune.py는 에포크마다 best weight를 RAM에 저장하고, "
         "훈련 루프 완전 종료 후에만 디스크에 씀. "
         "장점: 불필요한 디스크 I/O 없음. "
         "단점: 중간에 강제 종료(pkill)하면 best weight 소실. "
         "→ Phase 4 5차 시도에서 실제로 발생. "
         "개선 방향: ModelCheckpoint 콜백 추가 또는 일정 에포크마다 중간 저장."),
    ]

    for i, (title, content) in enumerate(decisions, 1):
        story.append(KeepTogether([
            Paragraph(f"11.{i}  {title}", S("h2")),
            Paragraph(content, S("body")),
        ]))

    # ═══════════════════════════════════════════════════════════════
    # 12. 저장된 파일
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("12.  저장된 모델 파일 및 TFLite 배포", S("h1")))

    files_data = [
        ["파일 경로", "크기", "저장 시각", "설명"],
        ["models/custom_classifier.h5", "7.9 MB", "09:23", "Phase 3 분류기 (embedding 입력)"],
        ["models/custom_classifier_finetuned.h5", "2.6 MB", "11:47", "Phase 4 파인튜닝 완료 분류기"],
        ["models/yamnet_finetuned/yamnet_vars.npz", "14.2 MB", "11:47", "Phase 4 YAMNet 가중치 (33개 변수)"],
        ["models/adas_detector.tflite", "0.67 MB", "11:48", "★ 배포용 INT8 양자화 TFLite"],
        ["data/processed/embeddings/embeddings.npz", "~34 MB", "09:22", "Phase 2 임베딩 캐시 (8,790×1,024)"],
    ]
    extra_f = [("BACKGROUND", (0,4), (-1,4), C_LGREEN), ("FONTNAME", (0,4), (-1,4), "MgB")]
    story.append(tbl(files_data, [W_PAGE*0.39, W_PAGE*0.10, W_PAGE*0.11, W_PAGE*0.40], extra_cmds=extra_f))
    story.append(Paragraph("표 12-1. 최종 저장 파일 목록 (2026-05-21 기준)", S("caption")))

    story.append(Paragraph("TFLite INT8 양자화 상세", S("h3")))
    story.append(Paragraph(
        "- 입력: yamnet_embedding (None, 1024) float32 → INT8 변환\n"
        "- 출력: 3-class 확률 (None, 3) → INT8 변환\n"
        "- 크기: 2.6 MB (float32 Keras) → 0.67 MB (int8 TFLite), 74% 감소\n"
        "- 목표 5 MB 이하 대비 6.7배 여유\n"
        "- Raspberry Pi 4B (ARM Cortex-A72) 호환",
        S("code")
    ))

    story.append(Paragraph("라즈베리파이 배포 절차", S("h3")))
    story.append(Paragraph(
        "# 1. 파일 복사\nscp models/adas_detector.tflite pi@raspberrypi:/home/pi/adas/models/\nscp config.yaml pi@raspberrypi:/home/pi/adas/\n\n"
        "# 2. 의존성 설치\npip install tflite-runtime sounddevice numpy PyYAML\n\n"
        "# 3. 실행\ncd /home/pi/adas\npython3 -m src.inference --config config.yaml",
        S("code")
    ))

    # ═══════════════════════════════════════════════════════════════
    # 13. 권장 후속 조치
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("13.  권장 후속 조치", S("h1")))

    next_steps = [
        ("최우선: 실제 Siren Recall 측정",
         "사이렌 음원 파일 10~20개 준비 → src/inference 실행 중 재생 → "
         "threshold=0.20 기준 실제 감지율 측정. "
         "목표 90% 미달 시 threshold를 0.15로 낮추거나 "
         "추가 사이렌 데이터로 Phase 4 재학습."),
        ("ModelCheckpoint 추가 (코드 개선)",
         "finetune.py에 일정 N 에포크마다 디스크 저장 기능 추가. "
         "현재는 강제 종료 시 가중치 소실 위험이 있음. "
         "예: patience=10 중간인 Epoch 5마다 임시 저장."),
        ("추가 데이터 확보 (siren)",
         "현재 사이렌 학습 데이터: UrbanSound8K 929개 × 5 SNR = ~4,600개. "
         "다양한 국가/차종의 사이렌 소리를 추가하면 일반화 성능 향상. "
         "freesound.org 또는 AudioSet에서 추가 수집 가능."),
        ("실차 테스트",
         "실제 도로 주행 중 긴급차량 근접 시나리오 테스트. "
         "차량 창문 개폐, 라디오 켜진 상태 등 다양한 조건에서 검증."),
        ("차량 내 소음 대응",
         "엔진 소음, 에어컨, 바람 소리 등이 background로 올바르게 분류되는지 확인. "
         "필요시 차량 내부 소음 데이터를 background에 추가."),
    ]
    for i, (title, content) in enumerate(next_steps, 1):
        story.append(KeepTogether([
            Paragraph(f"13.{i}  {title}", S("h3")),
            Paragraph(content, S("body")),
        ]))

    # ═══════════════════════════════════════════════════════════════
    # 14. 전체 요약
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("14.  전체 요약", S("h1")))
    story.append(Paragraph(
        "2026년 5월 21일 하루 동안 진행된 ADAS 음향 감지 시스템 전체 학습 과정을 정리합니다.",
        S("body")
    ))

    summary_data = [
        ["구분", "내용"],
        ["프로젝트 목적", "청각 장애 운전자를 위한 경적/사이렌 실시간 감지 경보 시스템"],
        ["모델 구조", "YAMNet(60% 파인튜닝) + Custom DNN(1024→512→256→64→3)"],
        ["데이터", "UrbanSound8K + STRAFFIC 도로 소음 + SNR 증강 (총 ~7,246 train)"],
        ["학습 단계", "Phase 0(CPU 베이스라인) → 1(YAMNet FT R1) → 2(임베딩 재추출) → 3(DNN) → 4(YAMNet FT R2)"],
        ["주요 시행착오", "GPU LD_LIBRARY_PATH 미설정 (2회)\nvenv 미활성화 ModuleNotFoundError (2회)\n강제 종료로 가중치 소실 (1회)"],
        ["최고 성능 (val)", "Epoch 30: val_loss=0.0065, val_acc=99.37%"],
        ["최종 test 성능", "car_horn R=97.0% ✅  siren R=68.7%(argmax) ⚠️  background R=97.0% ✅"],
        ["배포 파일", "adas_detector.tflite (INT8, 0.67 MB)"],
        ["핵심 교훈",
         "1. GPU 학습 시 LD_LIBRARY_PATH를 반드시 설정\n"
         "2. venv 활성화 확인 필수\n"
         "3. finetune은 early stopping 완료 전 강제 종료 금지\n"
         "4. argmax recall ≠ 실제 recall (threshold 방식이 더 높음)\n"
         "5. 검증 accuracy가 높아도 테스트셋 분포 차이로 낮을 수 있음"],
    ]
    extra_sum = [
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("BACKGROUND", (0,6), (-1,6), C_LGREEN),
        ("BACKGROUND", (0,8), (-1,8), C_LBLUE),
    ]
    story.append(tbl(summary_data, [W_PAGE*0.25, W_PAGE*0.75], extra_cmds=extra_sum))
    story.append(Paragraph("표 14-1. 전체 프로젝트 요약", S("caption")))

    story += [
        Spacer(1, 8*mm),
        HR(1.5, C_NAVY),
        Spacer(1, 3*mm),
        Paragraph(
            "ADAS Sound Detector — 종합 프로젝트 보고서<br/>"
            "작성일: 2026-05-21  |  Phase 0 ~ Phase 4 전 과정 기록",
            S("meta", fontSize=9)
        ),
    ]

    doc.build(story)
    print(f"\n✅ PDF 생성 완료: {OUTPUT}")
    import os
    size = os.path.getsize(OUTPUT)
    print(f"   파일 크기: {size/1024:.1f} KB")


if __name__ == "__main__":
    build()
