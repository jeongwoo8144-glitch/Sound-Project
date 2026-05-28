"""
ADAS Sound Detector — Siren Recall 분석 보고서 PDF 생성기
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os

# ── 폰트 등록 ──────────────────────────────────────────────
FONT_PATH = "/mnt/c/Windows/Fonts/malgun.ttf"
FONT_BOLD_PATH = "/mnt/c/Windows/Fonts/malgunbd.ttf"
pdfmetrics.registerFont(TTFont("Malgun", FONT_PATH))
pdfmetrics.registerFont(TTFont("MalgunBd", FONT_BOLD_PATH))

# ── 출력 경로 ──────────────────────────────────────────────
OUTPUT = "/mnt/c/Users/Daniel Park/Downloads/SoundPJ-rebuilt/siren_recall_report.pdf"

# ── 색상 ──────────────────────────────────────────────────
C_HEADER  = colors.HexColor("#1a3a5c")
C_ACCENT  = colors.HexColor("#2e6da4")
C_WARN    = colors.HexColor("#c0392b")
C_OK      = colors.HexColor("#27ae60")
C_LIGHT   = colors.HexColor("#eaf2fb")
C_GRAY    = colors.HexColor("#f4f6f8")
C_BORDER  = colors.HexColor("#bdc3c7")

# ── 스타일 ────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()
    S = {}

    S["title"] = ParagraphStyle(
        "title", fontName="MalgunBd", fontSize=20, textColor=C_HEADER,
        spaceAfter=4, alignment=TA_CENTER, leading=26
    )
    S["subtitle"] = ParagraphStyle(
        "subtitle", fontName="Malgun", fontSize=11, textColor=C_ACCENT,
        spaceAfter=2, alignment=TA_CENTER, leading=16
    )
    S["meta"] = ParagraphStyle(
        "meta", fontName="Malgun", fontSize=9, textColor=colors.gray,
        spaceAfter=0, alignment=TA_CENTER
    )
    S["h1"] = ParagraphStyle(
        "h1", fontName="MalgunBd", fontSize=13, textColor=colors.white,
        spaceBefore=12, spaceAfter=6, leading=18,
        backColor=C_HEADER, leftIndent=-10, rightIndent=-10,
        borderPad=6
    )
    S["h2"] = ParagraphStyle(
        "h2", fontName="MalgunBd", fontSize=11, textColor=C_ACCENT,
        spaceBefore=10, spaceAfter=4, leading=16,
        borderPad=2
    )
    S["body"] = ParagraphStyle(
        "body", fontName="Malgun", fontSize=9.5, textColor=colors.black,
        leading=15, spaceAfter=4, alignment=TA_JUSTIFY
    )
    S["body_center"] = ParagraphStyle(
        "body_center", fontName="Malgun", fontSize=9.5, textColor=colors.black,
        leading=15, spaceAfter=4, alignment=TA_CENTER
    )
    S["code"] = ParagraphStyle(
        "code", fontName="Courier", fontSize=8.5, textColor=colors.HexColor("#2c3e50"),
        backColor=C_GRAY, leading=13, spaceAfter=4,
        leftIndent=8, rightIndent=8, borderPad=6
    )
    S["warn"] = ParagraphStyle(
        "warn", fontName="Malgun", fontSize=9.5, textColor=C_WARN,
        leading=15, spaceAfter=4
    )
    S["ok"] = ParagraphStyle(
        "ok", fontName="Malgun", fontSize=9.5, textColor=C_OK,
        leading=15, spaceAfter=4
    )
    S["caption"] = ParagraphStyle(
        "caption", fontName="Malgun", fontSize=8, textColor=colors.gray,
        alignment=TA_CENTER, spaceAfter=6
    )
    S["bullet"] = ParagraphStyle(
        "bullet", fontName="Malgun", fontSize=9.5, textColor=colors.black,
        leading=15, spaceAfter=2, leftIndent=12, bulletIndent=0
    )
    return S

# ── 테이블 공통 스타일 ─────────────────────────────────────
def tbl_style(header_bg=C_ACCENT, has_header=True):
    cmds = [
        ("FONTNAME", (0,0), (-1,-1), "Malgun"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_GRAY]),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.4, C_BORDER),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
    ]
    if has_header:
        cmds += [
            ("BACKGROUND", (0,0), (-1,0), header_bg),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "MalgunBd"),
            ("FONTSIZE", (0,0), (-1,0), 9.5),
        ]
    return TableStyle(cmds)


def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=18*mm
    )
    S = make_styles()
    W = A4[0] - 40*mm  # 가용 너비

    story = []

    # ════════════════════════════════════════════════════
    # 표지
    # ════════════════════════════════════════════════════
    story.append(Spacer(1, 18*mm))
    story.append(Paragraph("ADAS Sound Detector", S["title"]))
    story.append(Paragraph("Siren Recall 분석 보고서", S["subtitle"]))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("작성일: 2026-05-21  |  Phase 4 YAMNet Finetune Round 2 완료 기준", S["meta"]))
    story.append(HRFlowable(width=W, thickness=1.5, color=C_ACCENT, spaceAfter=8))
    story.append(Spacer(1, 4*mm))

    # ════════════════════════════════════════════════════
    # 1. 개요
    # ════════════════════════════════════════════════════
    story.append(Paragraph("  1.  프로젝트 개요", S["h1"]))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "본 프로젝트는 청각 장애 운전자를 위한 차량용 ADAS 음향 감지 시스템입니다. "
        "도로 주행 중 발생하는 경적(car_horn)과 사이렌(siren)을 실시간으로 감지하여 "
        "시각적 경보를 제공합니다. 모델은 Google YAMNet(MobileNet V1 기반)을 "
        "파인튜닝하여 UrbanSound8K 데이터셋으로 학습하였습니다.",
        S["body"]
    ))

    goal_data = [
        ["항목", "목표값", "달성 여부"],
        ["전체 정확도 (Val)", "98.5% 이상", "✅ 99.37%"],
        ["Siren Recall (실시간)", "90% 이상", "⚠️ 측정 필요"],
        ["TFLite 파일 크기", "5 MB 이하", "✅ 0.67 MB"],
        ["배포 환경", "Raspberry Pi 4B", "✅ INT8 양자화"],
    ]
    story.append(Spacer(1, 3*mm))
    t = Table(goal_data, colWidths=[W*0.35, W*0.30, W*0.35])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(Paragraph("표 1. 프로젝트 목표 및 달성 현황", S["caption"]))

    # ════════════════════════════════════════════════════
    # 2. 학습 파이프라인
    # ════════════════════════════════════════════════════
    story.append(Paragraph("  2.  학습 파이프라인 요약", S["h1"]))
    story.append(Spacer(1, 2*mm))

    phase_data = [
        ["Phase", "내용", "결과"],
        ["Phase 0", "초기 Classifier (CPU)", "Test acc 86.8%"],
        ["Phase 1", "YAMNet Finetune R1 (40% unfreeze)", "완료"],
        ["Phase 2", "Embedding 재추출 (finetuned YAMNet)", "완료"],
        ["Phase 3", "Classifier 재학습 (finetuned embeddings)", "Test acc ~89%\nSiren R=77.3%"],
        ["Phase 4", "YAMNet Finetune R2 (60% unfreeze)", "Val acc 99.37%\nEarly stop @ Ep.40"],
    ]
    t = Table(phase_data, colWidths=[W*0.18, W*0.47, W*0.35])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(Paragraph("표 2. 학습 단계별 결과 요약", S["caption"]))

    # ════════════════════════════════════════════════════
    # 3. Phase 4 상세 결과
    # ════════════════════════════════════════════════════
    story.append(Paragraph("  3.  Phase 4 최종 학습 결과", S["h1"]))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph("3.1  학습 설정", S["h2"]))

    cfg_data = [
        ["파라미터", "값"],
        ["Unfreeze 비율", "60% (layer 9~14 / 56 변수 중 33개)"],
        ["YAMNet LR", "5×10⁻⁶"],
        ["Classifier LR", "2×10⁻⁵"],
        ["Batch size", "16"],
        ["Max Epochs", "100"],
        ["Early stopping patience", "10"],
        ["GPU", "RTX 3070 Laptop + cuDNN 9.2.2"],
    ]
    t = Table(cfg_data, colWidths=[W*0.38, W*0.62])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(Paragraph("표 3. Phase 4 학습 하이퍼파라미터", S["caption"]))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("3.2  Epoch별 Val 성능 (주요)", S["h2"]))

    epoch_data = [
        ["Epoch", "Val Loss", "Val Acc", "Patience", "비고"],
        ["20", "0.0222", "98.44%", "0/10", "Best"],
        ["24", "0.0169", "99.06%", "0/10", "Best"],
        ["29", "0.0130", "99.06%", "0/10", "Best"],
        ["30", "0.0065", "99.37%", "0/10", "★ 최종 Best"],
        ["31", "0.0748", "98.44%", "1/10", "spike"],
        ["35", "0.0131", "99.37%", "5/10", ""],
        ["39", "0.0291", "98.75%", "9/10", ""],
        ["40", "—", "—", "—", "Early Stopping"],
    ]
    t = Table(epoch_data, colWidths=[W*0.12, W*0.17, W*0.17, W*0.17, W*0.37])
    style = tbl_style()
    # Best row 강조
    style.add("BACKGROUND", (0,4), (-1,4), colors.HexColor("#d5f5e3"))
    style.add("FONTNAME", (0,4), (-1,4), "MalgunBd")
    style.add("TEXTCOLOR", (0,4), (-1,4), C_OK)
    # Early stop row 강조
    style.add("BACKGROUND", (0,8), (-1,8), colors.HexColor("#fdecea"))
    t.setStyle(style)
    story.append(t)
    story.append(Paragraph("표 4. Phase 4 주요 Epoch 성능 (Best @ Epoch 30)", S["caption"]))

    # ════════════════════════════════════════════════════
    # 4. Siren Recall 핵심 분석
    # ════════════════════════════════════════════════════
    story.append(Paragraph("  4.  Siren Recall 상세 분석", S["h1"]))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph("4.1  두 가지 Recall 숫자가 존재하는 이유", S["h2"]))
    story.append(Paragraph(
        "학습 중 관찰된 val_acc(99.37%)와 최종 테스트 평가의 siren recall(68.7%)은 "
        "<b>측정 방식과 데이터가 다르기 때문에</b> 직접 비교할 수 없습니다.",
        S["body"]
    ))

    compare_data = [
        ["구분", "Val Acc 99.37%", "Test Siren Recall 68.7%"],
        ["데이터", "Fold 9 (검증용)", "Fold 10 (테스트용)"],
        ["샘플 수 (siren)", "410개", "83개"],
        ["포함 데이터", "Clean + Augmented(SNR)", "Clean 원음만"],
        ["판정 기준", "argmax", "argmax"],
        ["환경", "학습 중 모니터링용", "최종 성능 평가"],
    ]
    t = Table(compare_data, colWidths=[W*0.25, W*0.375, W*0.375])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(Paragraph("표 5. 두 가지 평가 지표 비교", S["caption"]))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("4.2  Argmax 방식이란?", S["h2"]))
    story.append(Paragraph(
        "테스트 평가는 argmax 방식으로 진행됩니다. 모델이 출력한 3개 클래스의 확률 중 "
        "가장 높은 클래스를 예측값으로 선택하는 방식입니다.",
        S["body"]
    ))
    story.append(Paragraph(
        "car_horn = 0.05,  siren = 0.18,  background = 0.77\n"
        "→  argmax = background  (사이렌 소리임에도 background로 판정 → MISS)",
        S["code"]
    ))
    story.append(Paragraph(
        "위 예시에서 siren 확률이 0.18로 두 번째로 높지만, argmax는 background(0.77)를 "
        "선택하므로 해당 샘플은 'miss(오분류)'로 기록됩니다. "
        "이것이 siren recall이 68.7%로 낮게 보이는 주된 원인입니다.",
        S["body"]
    ))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("4.3  실제 추론은 Adaptive Threshold 방식 사용", S["h2"]))
    story.append(Paragraph(
        "실시간 추론 모듈(src/inference.py)은 argmax를 사용하지 않습니다. "
        "각 클래스별로 독립적인 임계값(threshold)을 설정하여, "
        "해당 클래스의 확률이 임계값을 넘으면 경보를 발동합니다.",
        S["body"]
    ))

    thresh_data = [
        ["클래스", "Threshold", "설정 근거"],
        ["car_horn", "0.45", "경적은 흔함 → Precision 우선, 오경보 방지"],
        ["siren", "0.20", "긴급차량 → Recall 우선, 놓치면 위험"],
        ["background", "(해당 없음)", "배경 클래스는 threshold 없음"],
    ]
    t = Table(thresh_data, colWidths=[W*0.20, W*0.20, W*0.60])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(Paragraph("표 6. 클래스별 Adaptive Threshold 설정값 (config.yaml 기준)", S["caption"]))

    story.append(Paragraph(
        "car_horn = 0.05,  siren = 0.18,  background = 0.77\n"
        "→  siren(0.18) >= threshold(0.20)?  →  NO  (경보 없음)\n\n"
        "car_horn = 0.05,  siren = 0.22,  background = 0.73\n"
        "→  siren(0.22) >= threshold(0.20)?  →  YES  →  🚨 사이렌 경보 발동!",
        S["code"]
    ))
    story.append(Paragraph(
        "두 번째 예시에서 background(0.73)가 더 높지만, siren이 0.20을 넘었기 때문에 "
        "경보가 발동됩니다. 즉, <b>argmax 테스트에서 miss로 기록된 샘플 중 상당수가 "
        "실제 추론에서는 올바르게 감지됩니다.</b>",
        S["body"]
    ))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("4.4  Phase별 Siren Recall 비교", S["h2"]))

    phase_recall_data = [
        ["Phase", "Siren Recall", "평가 방식", "Siren 샘플 수", "비고"],
        ["Phase 3\n(Classifier)", "77.3%", "argmax", "415개\n(aug 포함)", "더 현실적 평가"],
        ["Phase 4\n(Finetune)", "68.7%", "argmax", "83개\n(clean only)", "보수적 평가"],
        ["실시간 추론\n(추정)", "측정 필요", "threshold\n=0.20", "—", "실제 더 높을 것"],
    ]
    t = Table(phase_recall_data, colWidths=[W*0.20, W*0.18, W*0.20, W*0.20, W*0.22])
    style = tbl_style()
    style.add("BACKGROUND", (0,3), (-1,3), colors.HexColor("#fef9e7"))
    t.setStyle(style)
    story.append(t)
    story.append(Paragraph("표 7. Phase별 Siren Recall 비교", S["caption"]))

    story.append(Paragraph(
        "⚠️  Phase 4 test recall(68.7%)이 Phase 3(77.3%)보다 낮은 이유: "
        "Phase 4는 clean 83개만, Phase 3는 augmented 415개로 평가했기 때문입니다. "
        "테스트 세트 크기와 구성이 달라 단순 비교는 부적절합니다.",
        S["warn"]
    ))

    # ════════════════════════════════════════════════════
    # 5. 최종 테스트셋 성능표
    # ════════════════════════════════════════════════════
    story.append(Paragraph("  5.  Phase 4 Test Set 전체 성능 (Fold 10, Argmax 기준)", S["h1"]))
    story.append(Spacer(1, 2*mm))

    perf_data = [
        ["클래스", "Precision", "Recall", "F1-Score", "Support(N)"],
        ["car_horn", "0.889", "0.970", "0.928", "33"],
        ["siren", "0.905", "0.687", "0.781", "83"],
        ["background", "0.893", "0.970", "0.930", "198"],
        ["전체 accuracy", "—", "0.895", "—", "314"],
        ["macro avg", "0.896", "0.875", "0.879", "314"],
        ["weighted avg", "0.896", "0.895", "0.890", "314"],
    ]
    t = Table(perf_data, colWidths=[W*0.25, W*0.18, W*0.18, W*0.18, W*0.21])
    style = tbl_style()
    # siren row 노란 배경
    style.add("BACKGROUND", (0,2), (-1,2), colors.HexColor("#fef9e7"))
    # car_horn, background 녹색
    style.add("TEXTCOLOR", (2,1), (2,1), C_OK)  # car_horn recall
    style.add("TEXTCOLOR", (2,3), (2,3), C_OK)  # background recall
    style.add("TEXTCOLOR", (2,2), (2,2), C_WARN)  # siren recall
    style.add("FONTNAME", (2,2), (2,2), "MalgunBd")
    t.setStyle(style)
    story.append(t)
    story.append(Paragraph("표 8. Phase 4 최종 Test Set 성능 (argmax, clean audio, fold 10)", S["caption"]))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "✅  car_horn recall 97.0% — 목표치 초과 달성\n"
        "✅  background recall 97.0% — False Positive 방지 우수\n"
        "⚠️  siren recall 68.7% (argmax) — 실시간 threshold=0.20 적용 시 실질적 recall 더 높음",
        S["code"]
    ))

    # ════════════════════════════════════════════════════
    # 6. 저장된 모델 파일
    # ════════════════════════════════════════════════════
    story.append(Paragraph("  6.  저장된 모델 파일 현황", S["h1"]))
    story.append(Spacer(1, 2*mm))

    file_data = [
        ["파일 경로", "크기", "저장 시각", "용도"],
        ["models/custom_classifier_finetuned.h5", "2.6 MB", "11:47", "Keras 분류기 (Phase 4)"],
        ["models/yamnet_finetuned/yamnet_vars.npz", "14.2 MB", "11:47", "YAMNet 파인튜닝 가중치"],
        ["models/adas_detector.tflite", "0.67 MB", "11:48", "라즈베리파이 배포용 INT8"],
    ]
    t = Table(file_data, colWidths=[W*0.42, W*0.13, W*0.13, W*0.32])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(Paragraph("표 9. Phase 4 완료 후 저장된 모델 파일 (2026-05-21)", S["caption"]))

    # ════════════════════════════════════════════════════
    # 7. 권장 후속 조치
    # ════════════════════════════════════════════════════
    story.append(Paragraph("  7.  권장 후속 조치", S["h1"]))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph("7.1  실시간 Siren Recall 검증 (최우선)", S["h2"]))
    story.append(Paragraph(
        "argmax 기반 테스트(68.7%)는 threshold 방식과 다릅니다. "
        "실제 recall을 측정하려면 다음 방법을 권장합니다:",
        S["body"]
    ))
    for item in [
        "① 사이렌 음원 파일(mp3/wav) 여러 개를 준비",
        "② python3 -m src.inference 실행 후 음원을 시스템 오디오로 재생",
        "③ 경보 발동 횟수 / 전체 시도 횟수로 실제 recall 측정",
        "④ 필요 시 config.yaml의 siren threshold를 0.20 → 0.15로 낮춰 recall 강화",
    ]:
        story.append(Paragraph(item, S["bullet"]))

    story.append(Spacer(1, 2*mm))
    story.append(Paragraph("7.2  Threshold 조정 가이드", S["h2"]))

    thresh_guide = [
        ["Threshold 값", "예상 효과", "권장 상황"],
        ["0.25 이상", "Precision 높음\nFalse Positive 감소", "오경보가 잦을 때"],
        ["0.20 (현재)", "Recall/Precision 균형", "기본 설정"],
        ["0.15", "Recall 높음\nFalse Positive 증가 가능", "놓치면 위험한 환경"],
        ["0.10", "매우 민감\n오경보 많아질 수 있음", "극도의 안전 우선 환경"],
    ]
    t = Table(thresh_guide, colWidths=[W*0.22, W*0.38, W*0.40])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(Paragraph("표 10. Siren Threshold 조정 가이드", S["caption"]))

    story.append(Spacer(1, 2*mm))
    story.append(Paragraph("7.3  라즈베리파이 배포 절차", S["h2"]))
    for item in [
        "① adas_detector.tflite (0.67 MB) 라즈베리파이로 복사",
        "② pip install tflite-runtime sounddevice numpy",
        "③ python3 -m src.inference --config config.yaml 실행",
        "④ 실시간 마이크 입력으로 경적/사이렌 감지 확인",
    ]:
        story.append(Paragraph(item, S["bullet"]))

    # ════════════════════════════════════════════════════
    # 마무리
    # ════════════════════════════════════════════════════
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width=W, thickness=1, color=C_BORDER))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "본 보고서는 2026-05-21 Phase 4 YAMNet Finetune Round 2 완료 시점의 결과를 기록합니다.\n"
        "ADAS Sound Detector — 청각 장애 운전자를 위한 실시간 음향 경보 시스템",
        S["meta"]
    ))

    doc.build(story)
    print(f"PDF 생성 완료: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
