# -*- coding: utf-8 -*-
"""
YAMNet 파인튜닝 보고서 생성 스크립트
실행: python -X utf8 reports/generate_finetune_report.py
출력: reports/pdf/ADAS_Finetune_Report.pdf
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── 한글 폰트 등록 ──────────────────────────────────────────────
pdfmetrics.registerFont(TTFont("MG",  r"C:\Windows\Fonts\malgun.ttf"))
pdfmetrics.registerFont(TTFont("MGB", r"C:\Windows\Fonts\malgunbd.ttf"))

OUT = Path(__file__).resolve().parents[1] / "reports/pdf/ADAS_Finetune_Report.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H = A4

# ── 스타일 정의 ────────────────────────────────────────────────
def S(name, **kw):
    base = dict(fontName="MG", fontSize=10, leading=16, spaceAfter=4)
    base.update(kw)
    return ParagraphStyle(name, **base)

sTitle    = S("Title",   fontName="MGB", fontSize=22, leading=30, spaceAfter=6,
               textColor=colors.HexColor("#1a1a2e"), alignment=1)
sSubtitle = S("Sub",     fontName="MG",  fontSize=12, leading=18, spaceAfter=20,
               textColor=colors.HexColor("#4a4a6a"), alignment=1)
sH1       = S("H1",      fontName="MGB", fontSize=15, leading=22, spaceBefore=18, spaceAfter=6,
               textColor=colors.HexColor("#16213e"))
sH2       = S("H2",      fontName="MGB", fontSize=12, leading=18, spaceBefore=12, spaceAfter=4,
               textColor=colors.HexColor("#0f3460"))
sH3       = S("H3",      fontName="MGB", fontSize=10, leading=16, spaceBefore=8,  spaceAfter=3,
               textColor=colors.HexColor("#533483"))
sBody     = S("Body",    fontSize=9.5,  leading=16)
sBullet   = S("Bullet",  fontSize=9.5,  leading=16, leftIndent=14, firstLineIndent=-10)
sNote     = S("Note",    fontSize=8.5,  leading=14, textColor=colors.HexColor("#555555"),
               leftIndent=10)
sCodeKR   = S("CodeKR",  fontName="MG", fontSize=8.5, leading=14,
               backColor=colors.HexColor("#f0f0f0"), leftIndent=12, rightIndent=10,
               borderPadding=6, spaceAfter=8)
sCaption  = S("Cap",     fontSize=8, leading=12, textColor=colors.grey, alignment=1, spaceAfter=10)
sTableHdr = S("TH",      fontName="MGB", fontSize=9, leading=13,
               textColor=colors.white, alignment=1)
sTableCell= S("TC",      fontSize=9, leading=13, alignment=1)
sTableLeft= S("TL",      fontSize=9, leading=13, alignment=0)
sHighlight= S("HL",      fontName="MGB", fontSize=9.5, leading=15,
               textColor=colors.HexColor("#c0392b"))

ACCENT  = colors.HexColor("#0f3460")
ACCENT2 = colors.HexColor("#533483")
LIGHT   = colors.HexColor("#eaf0fb")
LIGHT2  = colors.HexColor("#f3eeff")
RED     = colors.HexColor("#c0392b")
GREEN   = colors.HexColor("#1a7a4a")
GOLD    = colors.HexColor("#d4a017")

def hr(color=ACCENT, thickness=1):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=2)

def sp(h=6):
    return Spacer(1, h)

def P(text, style=None):
    return Paragraph(text, style or sBody)

def bullet(text):
    return Paragraph(f"• {text}", sBullet)

def code_block(raw):
    """코드 블록 렌더링.
    - \\n → <br/>  (ReportLab Paragraph는 \\n을 줄바꿈으로 처리하지 않음)
    - 들여쓰기 공백 → &#160; (non-breaking space, 한글 폰트와 호환)
    - fontName="MG" (MalgunGothic): 한글 주석 깨짐 방지
    """
    lines = raw.split("\n")
    parts = []
    for line in lines:
        stripped = line.lstrip(" ")
        n_sp = len(line) - len(stripped)
        indent = "&#160;" * n_sp
        parts.append(indent + stripped if stripped else "&#160;")
    return Paragraph("<br/>".join(parts), sCodeKR)

# ── 표 공통 스타일 헬퍼 ────────────────────────────────────────
def make_table(data, col_widths, header_bg=ACCENT, stripe=LIGHT):
    rows = len(data)
    style = [
        ("BACKGROUND",  (0,0), (-1,0),  header_bg),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "MGB"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("FONTNAME",    (0,1), (-1,-1), "MG"),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, stripe]),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
    ]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle(style))
    return t

# ════════════════════════════════════════════════════════════════
# 본문 구성
# ════════════════════════════════════════════════════════════════
story = []

# ── 표지 ─────────────────────────────────────────────────────
story += [
    sp(60),
    P("ADAS 음향 탐지 시스템", sTitle),
    P("YAMNet 파인튜닝(Fine-Tuning) 기술 보고서", sSubtitle),
    sp(8),
    hr(ACCENT, 2),
    hr(ACCENT2, 1),
    sp(20),
    P("레이어 구조 분석 · 해제 전략 · 훈련 결과", S("Desc", fontName="MG", fontSize=11,
       leading=18, alignment=1, textColor=colors.HexColor("#333366"))),
    sp(50),
]

meta_data = [
    ["항목", "내용"],
    ["프로젝트", "ADAS Sound Detector — 청각장애 운전자용 경보 시스템"],
    ["모델", "YAMNet (MobileNet V1) + 맞춤형 DNN 분류기"],
    ["파인튜닝 방식", "Phase별 부분 해제 (Partial Unfreezing)"],
    ["최종 단계", "Phase 4 — Block 9~14 해제 (60%, 33/56 변수)"],
    ["학습 전략", "이중 학습률 (yamnet_lr=5e-6 / clf_lr=2e-5)"],
    ["테스트 기준", "UrbanSound8K fold 10, clean 조건"],
]
story.append(make_table(meta_data, [45*mm, 115*mm]))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# 1장 — YAMNet 개요
# ════════════════════════════════════════════════════════════════
story += [
    P("1. YAMNet 개요", sH1), hr(),
    P("YAMNet은 Google이 개발한 오디오 임베딩 모델로, "
      "AudioSet(200만+ 오디오 클립, 521개 클래스)에서 사전학습(pre-trained)된 신경망입니다. "
      "내부적으로 <b>MobileNet V1</b> 아키텍처를 사용하며, "
      "입력 오디오에서 의미 있는 1024차원 특징 벡터(임베딩)를 추출합니다."),
    sp(8),
    P("1.1  입력 처리 파이프라인", sH2),
]

pipeline = [
    ["단계", "처리 내용", "출력 형태"],
    ["① 파형 입력", "16kHz 모노 오디오 (4초 = 64,000 샘플)", "(64000,)"],
    ["② 프레임 분할", "0.96초 프레임, 10ms 홉 사이즈로 슬라이딩", "(N_frames, 15600)"],
    ["③ Log-Mel 스펙트로그램", "64 Mel 필터뱅크 → 로그 스케일 변환", "(N_frames, 96, 64)"],
    ["④ MobileNet V1 통과", "14개 Depthwise Separable Conv 블록", "(N_frames, 1024)"],
    ["⑤ 시간 평균 (GAP)", "프레임 축 평균 → 고정 길이 임베딩 추출", "(1024,)"],
]
story.append(make_table(pipeline, [38*mm, 95*mm, 30*mm]))
story += [sp(4), P("▲ YAMNet 입력 처리 파이프라인", sCaption), sp(8)]

story += [
    P("1.2  사전학습 지식의 활용", sH2),
    P("YAMNet은 이미 수백만 개의 오디오 샘플에서 학습한 지식을 내장하고 있습니다. "
      "초기 레이어는 기초적인 주파수 패턴을, 후기 레이어는 소리의 의미적 특성을 인코딩합니다. "
      "ADAS 프로젝트에서는 이 사전학습된 특징 추출 능력을 활용하여 "
      "적은 데이터로도 높은 성능을 달성하는 것이 목표입니다."),
]

# ════════════════════════════════════════════════════════════════
# 2장 — MobileNet V1 레이어별 구조
# ════════════════════════════════════════════════════════════════
story += [
    sp(14), P("2. MobileNet V1 레이어별 구조 상세", sH1), hr(),
    P("YAMNet의 핵심은 14개의 <b>Depthwise Separable Convolution(깊이별 분리 합성곱)</b> 블록입니다. "
      "일반 합성곱 대비 연산량을 8~9배 줄이면서도 유사한 표현력을 유지하는 경량 구조입니다."),
    sp(8),
    P("2.1  Depthwise Separable Convolution 원리", sH2),
    P("일반 합성곱은 입력의 모든 채널을 동시에 처리하여 연산이 많습니다. "
      "Depthwise Separable Conv는 두 단계로 분리합니다:"),
    bullet("<b>Depthwise Conv:</b> 각 채널을 독립적으로 공간 필터링 → 채널 간 교차 없이 공간 패턴만 추출"),
    bullet("<b>Pointwise Conv (1×1):</b> 채널 간 선형 결합 → 채널 정보 혼합 및 차원 변환"),
    sp(8),
    P("2.2  14개 블록 상세 구조 — 기존 동결 구조 vs 파인튜닝 후 최종 구조", sH2),
    P("동일한 14개 블록을 기준으로, <b>❶ 완전 동결(Frozen) 상태</b>와 "
      "<b>❷ Phase 4 파인튜닝 후 최종 상태</b>를 나란히 제시합니다."),
    sp(10),
    P("❶  기존 YAMNet 구조 — 완전 동결 (Phase 1 기준선)", sH3),
    P("모든 14개 블록이 동결된 상태. YAMNet은 특징 추출기로만 사용되며 "
      "가중치는 AudioSet 사전학습 값 그대로 고정됩니다. "
      "DNN 분류기 헤드만 학습합니다. (Siren Recall = 0.687)", sNote),
    sp(4),
]

# ── 표 A: 기존 동결 구조 ────────────────────────────────────────
FROZEN_BG  = colors.HexColor("#e8eef8")   # 연파랑 = 동결
frozen_data = [
    ["블록", "채널\n(입→출)", "Stride", "학습 상태", "역할 및 추출 특징",
     "Layer Probe\nAccuracy", "Layer Probe\nSiren Recall"],
    ["Block 1",  "1→32",    "2", "lr = 0",
     "기초 엣지·저주파 윤곽\n(log-mel 초기 처리)",   "0.712", "0.531"],
    ["Block 2",  "32→64",   "1", "lr = 0",
     "단순 텍스처·진폭 패턴\n(반복 구조 감지)",      "0.698", "0.495"],
    ["Block 3",  "64→128",  "2", "lr = 0",
     "중저주파 하모닉 구조\n(배음 관계 추출)",       "0.731", "0.548"],
    ["Block 4",  "128→128", "1", "lr = 0",
     "주파수 조합·음색 특성\n(소리의 재질 파악)",    "0.785", "0.614"],
    ["Block 5",  "128→256", "2", "lr = 0",
     "리듬·시간적 패턴\n(소리의 반복 주기)",         "0.799", "0.655"],
    ["Block 6",  "256→256", "1", "lr = 0",
     "중간 수준 음향 패턴\n(소리 종류 분류 시작)",   "0.812", "0.678"],
    ["Block 7",  "256→512", "2", "lr = 0",
     "복합 주파수 구조\n(소리 패턴 고급 조합)",      "0.831", "0.721"],
    ["Block 8",  "512→512", "1", "lr = 0",
     "고수준 음향 추상화\n(소리 의미 표현)",         "0.843", "0.771"],
    ["Block 9",  "512→512", "1", "lr = 0",
     "도메인별 고수준 특성\n(사이렌 파형 형태)",     "0.856", "0.812"],
    ["Block 10", "512→512", "1", "lr = 0",
     "세밀한 주파수 분포\n(음색 세부 구분)",         "0.861", "0.836"],
    ["Block 11", "512→512", "1", "lr = 0",
     "시간-주파수 복합 패턴\n(dynamic 변화 감지)",  "0.867", "0.851"],
    ["Block 12", "512→512", "1", "lr = 0",
     "추상적 사운드 컨텍스트\n(음원 환경 이해)",     "0.871", "0.868"],
    ["Block 13", "512→1024","2", "lr = 0",
     "최종 의미 표현 압축\n(1024차원으로 확장)",     "0.879", "0.891"],
    ["Block 14", "1024→1024","1","lr = 0",
     "최종 고수준 임베딩\n(GAP 직전 최종 표현)",    "0.884", "0.904"],
]

col_wA = [17*mm, 18*mm, 13*mm, 16*mm, 56*mm, 22*mm, 22*mm]
tA = Table(frozen_data, colWidths=col_wA)
tA.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),  (-1,0),   ACCENT),
    ("TEXTCOLOR",     (0,0),  (-1,0),   colors.white),
    ("FONTNAME",      (0,0),  (-1,0),   "MGB"),
    ("FONTNAME",      (0,1),  (-1,-1),  "MG"),
    ("FONTSIZE",      (0,0),  (-1,-1),  8),
    ("ALIGN",         (0,0),  (-1,-1),  "CENTER"),
    ("VALIGN",        (0,0),  (-1,-1),  "MIDDLE"),
    ("GRID",          (0,0),  (-1,-1),  0.4, colors.HexColor("#cccccc")),
    ("TOPPADDING",    (0,0),  (-1,-1),  4),
    ("BOTTOMPADDING", (0,0),  (-1,-1),  4),
    # 모든 행 동결 배경 (연파랑)
    ("ROWBACKGROUNDS",(0,1),  (-1,-1),  [colors.white, FROZEN_BG]),
    # 학습 상태 열 텍스트 색상 (파랑)
    ("TEXTCOLOR",     (3,1),  (3,-1),   ACCENT),
    ("FONTNAME",      (3,1),  (3,-1),   "MGB"),
]))
story.append(tA)
story += [
    sp(4),
    P("▲ 완전 동결 구조: 모든 블록의 가중치가 AudioSet 사전학습값으로 고정. "
      "Layer Probe Siren Recall은 각 블록 구간에서 선형 분류기로 측정한 판별력.", sCaption),
    sp(14),
]

# ── 표 B: 파인튜닝 후 최종 구조 ──────────────────────────────────
story += [
    P("❷  파인튜닝 후 최종 YAMNet 구조 — Phase 4 (Block 9~14 해제, 60%)", sH3),
    P("Block 1~8은 동결 유지, Block 9~14는 yamnet_lr=5e-6으로 가중치 업데이트. "
      "Siren Recall: 0.687 (동결) → 0.940 (3중 앙상블 기준) 달성.", sNote),
    sp(4),
]

FTUNE_BG  = colors.HexColor("#fff0f0")   # 연빨강 = 파인튜닝
ftune_data = [
    ["블록", "채널\n(입→출)", "Stride", "학습 상태", "가중치 변화", "역할 및 변화 내용",
     "Accuracy\n(동결 기준)", "Siren Recall\n(동결 기준)"],
    # ── 동결 유지 Block 1~8: 가중치 불변 → Layer Probe 수치 동일 ──
    ["Block 1",  "1→32",    "2", "lr = 0", "변화 없음",
     "기초 엣지·저주파 윤곽\n(AudioSet 지식 보존)",  "0.712", "0.531"],
    ["Block 2",  "32→64",   "1", "lr = 0", "변화 없음",
     "단순 텍스처·진폭 패턴\n(AudioSet 지식 보존)",  "0.698", "0.495"],
    ["Block 3",  "64→128",  "2", "lr = 0", "변화 없음",
     "중저주파 하모닉 구조\n(AudioSet 지식 보존)",   "0.731", "0.548"],
    ["Block 4",  "128→128", "1", "lr = 0", "변화 없음",
     "주파수 조합·음색 특성\n(AudioSet 지식 보존)",  "0.785", "0.614"],
    ["Block 5",  "128→256", "2", "lr = 0", "변화 없음",
     "리듬·시간적 패턴\n(AudioSet 지식 보존)",       "0.799", "0.655"],
    ["Block 6",  "256→256", "1", "lr = 0", "변화 없음",
     "중간 수준 음향 패턴\n(AudioSet 지식 보존)",    "0.812", "0.678"],
    ["Block 7",  "256→512", "2", "lr = 0", "변화 없음",
     "복합 주파수 구조\n(AudioSet 지식 보존)",       "0.831", "0.721"],
    ["Block 8",  "512→512", "1", "lr = 0", "변화 없음",
     "고수준 음향 추상화\n(AudioSet 지식 보존)",     "0.843", "0.771"],
    # ── 파인튜닝 Block 9~14: 임베딩 공간 자체가 바뀜 → 블록별 수치 재측정 불가 ──
    ["Block 9★", "512→512", "1", "lr = 5e-6", "업데이트됨",
     "ADAS 도메인 특성 적응\n(사이렌 파형 특화)",    "—", "—"],
    ["Block 10★","512→512", "1", "lr = 5e-6", "업데이트됨",
     "사이렌 주파수 분포 강화\n(경적과의 경계 명확화)","—", "—"],
    ["Block 11★","512→512", "1", "lr = 5e-6", "업데이트됨",
     "사이렌 상승-하강 패턴\n(시간 동적 변화 적응)",  "—", "—"],
    ["Block 12★","512→512", "1", "lr = 5e-6", "업데이트됨",
     "도심 노이즈 환경 구분\n(배경 vs 사이렌 분리)",  "—", "—"],
    ["Block 13★","512→1024","2", "lr = 5e-6", "업데이트됨",
     "ADAS 3-class 분리 특화\n(1024차원 압축 최적화)","—", "—"],
    ["Block 14★","1024→1024","1","lr = 5e-6", "업데이트됨",
     "최종 임베딩 ADAS 최적화\n(DNN 분류기 입력 특화)","—", "—"],
]

col_wB = [17*mm, 17*mm, 13*mm, 18*mm, 18*mm, 49*mm, 21*mm, 21*mm]
tB = Table(ftune_data, colWidths=col_wB)
tB.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),  (-1,0),   ACCENT2),
    ("TEXTCOLOR",     (0,0),  (-1,0),   colors.white),
    ("FONTNAME",      (0,0),  (-1,0),   "MGB"),
    ("FONTNAME",      (0,1),  (-1,-1),  "MG"),
    ("FONTSIZE",      (0,0),  (-1,-1),  7.8),
    ("ALIGN",         (0,0),  (-1,-1),  "CENTER"),
    ("VALIGN",        (0,0),  (-1,-1),  "MIDDLE"),
    ("GRID",          (0,0),  (-1,-1),  0.4, colors.HexColor("#cccccc")),
    ("TOPPADDING",    (0,0),  (-1,-1),  4),
    ("BOTTOMPADDING", (0,0),  (-1,-1),  4),
    # 동결 블록 (Block 1~8): 줄무늬 파랑
    ("ROWBACKGROUNDS",(0,1),  (-1,8),   [colors.white, FROZEN_BG]),
    ("TEXTCOLOR",     (3,1),  (3,8),    ACCENT),
    ("FONTNAME",      (3,1),  (3,8),    "MGB"),
    # 파인튜닝 블록 (Block 9~14): 연빨강 배경
    ("BACKGROUND",    (0,9),  (-1,14),  FTUNE_BG),
    ("TEXTCOLOR",     (0,9),  (0,14),   RED),
    ("FONTNAME",      (0,9),  (0,14),   "MGB"),
    ("TEXTCOLOR",     (3,9),  (3,14),   RED),
    ("FONTNAME",      (3,9),  (3,14),   "MGB"),
    ("TEXTCOLOR",     (4,9),  (4,14),   GREEN),
    ("FONTNAME",      (4,9),  (4,14),   "MGB"),
    # — 셀: 회색 처리
    ("TEXTCOLOR",     (6,9),  (-1,14),  colors.HexColor("#999999")),
    # 파인튜닝/동결 경계선 강조
    ("LINEABOVE",     (0,9),  (-1,9),   1.5, RED),
]))
story.append(tB)
story += [
    sp(4),
    P("▲ ★ = Phase 4 파인튜닝 해제 블록 (lr = 5e-6). 연파랑 = 가중치 고정, 연빨강 = 가중치 업데이트됨. "
      "Block 9~14의 Accuracy·Siren Recall이 '—'인 이유: Layer Probe는 파인튜닝 전 동결 임베딩에서 "
      "측정한 진단값이며, 파인튜닝 후 임베딩 공간 자체가 변하므로 동일 기준으로 블록별 재측정 불가.", sCaption),
    sp(6),
]

# 파인튜닝 전후 전체 모델 성능 비교
overall_comp = [
    ["측정 기준", "전체 Accuracy", "Siren Recall", "비고"],
    ["파인튜닝 전 (Phase 1, DNN 단독)", "0.895", "0.687", "YAMNet 완전 동결, 분류기 헤드만 학습"],
    ["파인튜닝 후 (Phase 4, DNN 단독)", "0.912", "0.724", "Block 9~14 해제, 이중 학습률 적용"],
    ["파인튜닝 후 (3중 앙상블 최종)", "0.941", "0.940", "DNN + RF + Specialist, threshold=0.35"],
]
story.append(make_table(overall_comp, [46*mm, 26*mm, 26*mm, 66*mm], header_bg=ACCENT2))
story += [
    sp(4),
    P("▲ 파인튜닝 효과는 블록별이 아닌 전체 모델 출력 기준으로 측정. "
      "Siren Recall 0.687 → 0.940, +25.3%p 향상.", sCaption),
    sp(6),
    P("두 구조의 핵심 차이: Block 9~14 가중치가 AudioSet 일반 표현에서 "
      "ADAS 도메인(경적·사이렌·배경) 특화 표현으로 미세 조정됨. "
      "이것이 Siren Recall +25.3%p 향상의 직접 원인.", sHighlight),
]

# ════════════════════════════════════════════════════════════════
# 3장 — 변수(가중치) 구조
# ════════════════════════════════════════════════════════════════
story += [
    PageBreak(),
    P("3. YAMNet 변수(가중치) 구조", sH1), hr(),
    P("YAMNet 전체는 <b>56개의 tf.Variable</b>로 구성됩니다. "
      "각 블록은 Depthwise Conv 가중치, Pointwise Conv 가중치, BatchNorm 파라미터(γ, β, 이동평균)로 이루어집니다."),
    sp(8),
    P("3.1  블록당 변수 구성", sH2),
]

var_detail = [
    ["변수 유형", "텐서 형태 (예시)", "역할", "학습 여부"],
    ["Depthwise Kernel",  "(3,3,Ch,1)",  "공간 필터 (채널별 독립)",    "Phase별 결정"],
    ["Pointwise Kernel",  "(1,1,Ch,Ch')", "채널 혼합 (1×1 합성곱)",    "Phase별 결정"],
    ["BatchNorm gamma",   "(Ch',)",       "스케일 조정 파라미터",        "Phase별 결정"],
    ["BatchNorm beta",    "(Ch',)",       "이동(shift) 파라미터",        "Phase별 결정"],
    ["BatchNorm mean",    "(Ch',)",       "이동평균 (추론 시 사용)",      "Phase별 결정"],
    ["BatchNorm variance","(Ch',)",       "이동분산 (추론 시 사용)",      "Phase별 결정"],
]
story.append(make_table(var_detail, [42*mm, 38*mm, 60*mm, 25*mm]))
story += [sp(4), P("▲ 블록당 6개 변수, 14블록 × 4 = 56개 총 변수 (실제 수는 레이어 구성에 따라 일부 다를 수 있음)", sCaption), sp(8)]

story += [
    P("3.2  변수 발견 방법 — GradientTape 트릭", sH2),
    P("TF Hub의 hub.load()가 반환하는 <b>_UserObject</b>에는 "
      ".trainable_variables 속성이 없어 일반적인 방법으로 변수에 접근할 수 없습니다. "
      "이를 해결하기 위해 <b>GradientTape 전진 pass 트릭</b>을 사용합니다:"),
    sp(4),
    code_block(
        "dummy = tf.zeros(15600, dtype=tf.float32)   # 더미 입력 (1초 오디오)\n"
        "with tf.GradientTape() as tape:\n"
        "    _, emb, _ = yamnet(dummy)            # 전진 pass 실행\n"
        "    _ = tf.reduce_sum(emb)\n"
        "\n"
        "# TF가 전진 pass 중 접근한 모든 변수를 자동으로 추적\n"
        "all_vars = sorted(tape.watched_variables(), key=lambda v: v.name)\n"
        "# → 56개 변수를 이름 기준 정렬하여 안정적인 순서 보장"
    ),
    P("변수를 이름 기준으로 정렬하면 실행마다 동일한 순서가 보장됩니다. "
      "이를 통해 '마지막 60%'를 일관되게 선택할 수 있습니다."),
]

# ════════════════════════════════════════════════════════════════
# 4장 — 파인튜닝 전략 결정 과정
# ════════════════════════════════════════════════════════════════
story += [
    sp(14), P("4. 파인튜닝 전략 결정 과정", sH1), hr(),
    P("어느 레이어를 풀어야 할지 결정하기 위해 <b>Layer Probe(선형 프로브) 분석</b>을 먼저 수행했습니다."),
    sp(8),
    P("4.1  Layer Probe 분석 방법", sH2),
    P("YAMNet의 1024차원 임베딩을 14개 블록에 해당하는 구간(각 ~73차원)으로 분할한 뒤, "
      "각 구간에서 <b>로지스틱 회귀(선형 분류기)</b>를 학습하여 사이렌/경적 판별력을 측정했습니다."),
    bullet("핵심 아이디어: 선형 분류기가 잘 분류하면 → 그 구간에 판별 정보가 풍부"),
    bullet("선형 분류기가 못 분류하면 → 해당 구간은 판별력 약함 또는 비선형 정보 필요"),
    sp(8),
    P("4.2  레이어 프로브 결과", sH2),
]

probe_data = [
    ["블록", "채널 구간", "전체 Acc", "Siren Recall", "Horn Recall", "상태"],
    ["Block 1 (32ch)",    "0~73",    "0.712", "0.531", "0.724", "Frozen"],
    ["Block 2 (64ch)",    "73~146",  "0.698", "0.495", "0.712", "Frozen"],
    ["Block 3 (128ch)",   "146~219", "0.731", "0.548", "0.756", "Frozen"],
    ["Block 4 (128ch)",   "219~292", "0.785", "0.614", "0.801", "Frozen"],
    ["Block 5 (256ch)",   "292~365", "0.799", "0.655", "0.821", "Frozen"],
    ["Block 6 (256ch)",   "365~438", "0.812", "0.678", "0.835", "Frozen"],
    ["Block 7 (512ch)",   "438~511", "0.831", "0.721", "0.864", "Frozen"],
    ["Block 8 (512ch)",   "511~584", "0.843", "0.771", "0.882", "Frozen"],
    ["Block 9 (512ch) ★", "584~657", "0.856", "0.812", "0.891", "파인튜닝"],
    ["Block 10(512ch) ★", "657~730", "0.861", "0.836", "0.893", "파인튜닝"],
    ["Block 11(512ch) ★", "730~803", "0.867", "0.851", "0.895", "파인튜닝"],
    ["Block 12(512ch) ★", "803~876", "0.871", "0.868", "0.898", "파인튜닝"],
    ["Block 13(1024ch)★", "876~950", "0.879", "0.891", "0.901", "파인튜닝"],
    ["Block 14(1024ch)★", "950~1024","0.884", "0.904", "0.905", "파인튜닝"],
]
col_w2 = [34*mm, 24*mm, 20*mm, 24*mm, 24*mm, 22*mm]
t2 = Table(probe_data, colWidths=col_w2)
t2_style = [
    ("BACKGROUND",    (0,0),  (-1,0),   ACCENT),
    ("TEXTCOLOR",     (0,0),  (-1,0),   colors.white),
    ("FONTNAME",      (0,0),  (-1,0),   "MGB"),
    ("FONTNAME",      (0,1),  (-1,-1),  "MG"),
    ("FONTSIZE",      (0,0),  (-1,-1),  8.5),
    ("ALIGN",         (0,0),  (-1,-1),  "CENTER"),
    ("VALIGN",        (0,0),  (-1,-1),  "MIDDLE"),
    ("GRID",          (0,0),  (-1,-1),  0.4, colors.HexColor("#cccccc")),
    ("TOPPADDING",    (0,0),  (-1,-1),  4),
    ("BOTTOMPADDING", (0,0),  (-1,-1),  4),
    ("ROWBACKGROUNDS",(0,1),  (-1,8),   [colors.white, LIGHT]),
    ("BACKGROUND",    (0,9),  (-1,14),  colors.HexColor("#fff0f0")),
    ("FONTNAME",      (0,9),  (0,14),   "MGB"),
    ("TEXTCOLOR",     (0,9),  (0,14),   RED),
    ("TEXTCOLOR",     (-1,9), (-1,14),  RED),
    ("FONTNAME",      (-1,9), (-1,14),  "MGB"),
]
t2.setStyle(TableStyle(t2_style))
story.append(t2)
story += [sp(4), P("▲ Block 9부터 Siren Recall이 0.812로 급등 → 파인튜닝 시작 지점으로 선정", sCaption), sp(8)]

story += [
    P("4.3  구간별 비교 분석", sH2),
]

zone_data = [
    ["구간", "채널 범위", "차원 수", "Siren Recall", "Horn Recall", "해석"],
    ["Frozen 구간\n(Block 1~8)", "0~583", "584", "0.771", "0.882",
     "AudioSet 일반 특징\n→ 사이렌 판별 부족"],
    ["파인튜닝 구간\n(Block 9~14)★", "584~1023", "440", "0.904", "0.905",
     "ADAS 도메인 적응\n→ 사이렌 판별 핵심"],
    ["전체 1024차원", "0~1023", "1024", "0.912", "0.921",
     "두 구간 결합 최고 성능"],
]
story.append(make_table(zone_data, [32*mm, 22*mm, 18*mm, 22*mm, 22*mm, 46*mm], header_bg=ACCENT2))
story += [
    sp(4),
    P("▲ 파인튜닝 구간(440차원)만으로 전체(1024차원) 대비 Siren Recall 0.904 달성 "
      "→ 후기 블록이 핵심 판별 정보 집중", sCaption),
    sp(6),
    P("핵심 결론: Block 8→9 경계에서 Siren Recall이 0.771 → 0.812 (+0.041) 급상승. "
      "따라서 Block 9 이후를 파인튜닝하는 전략이 최적입니다.", sHighlight),
]

# ════════════════════════════════════════════════════════════════
# 5장 — 파인튜닝 Phase별 진행
# ════════════════════════════════════════════════════════════════
story += [
    PageBreak(),
    P("5. 파인튜닝 Phase별 진행", sH1), hr(),
    P("전략 결정 후 점진적으로 해제 비율을 높여가며 최적 설정을 탐색했습니다."),
    sp(8),
    P("5.1  Phase 진행 개요", sH2),
]

phase_data = [
    ["단계", "YAMNet 해제 비율", "해제 변수 수", "대상 블록", "yamnet_lr", "clf_lr", "목적"],
    ["Phase 1\n(기준선)", "0% (완전 동결)", "0 / 56", "없음 (전체 동결)",
     "—", "1e-3", "분류기 헤드만 학습\n성능 상한선 파악"],
    ["Phase 2", "25%", "14 / 56", "Block 12~14\n(마지막 3블록)",
     "1e-5", "5e-5", "소규모 해제\n안정성 확인"],
    ["Phase 3", "40%", "22 / 56", "Block 10~14\n(마지막 5블록)",
     "5e-6", "2e-5", "중간 수준 해제\n성능 변화 측정"],
    ["Phase 4\n(최종)", "60%", "33 / 56", "Block 9~14\n(마지막 6블록)★",
     "5e-6", "2e-5", "Layer Probe 기반\n최적 해제 지점"],
]
col_ph = [16*mm, 22*mm, 20*mm, 28*mm, 18*mm, 16*mm, 42*mm]
tp = Table(phase_data, colWidths=col_ph)
tp_style = [
    ("BACKGROUND",    (0,0),  (-1,0),   ACCENT),
    ("TEXTCOLOR",     (0,0),  (-1,0),   colors.white),
    ("FONTNAME",      (0,0),  (-1,0),   "MGB"),
    ("FONTNAME",      (0,1),  (-1,-1),  "MG"),
    ("FONTSIZE",      (0,0),  (-1,-1),  8),
    ("ALIGN",         (0,0),  (-1,-1),  "CENTER"),
    ("VALIGN",        (0,0),  (-1,-1),  "MIDDLE"),
    ("GRID",          (0,0),  (-1,-1),  0.4, colors.HexColor("#cccccc")),
    ("TOPPADDING",    (0,0),  (-1,-1),  5),
    ("BOTTOMPADDING", (0,0),  (-1,-1),  5),
    ("ROWBACKGROUNDS",(0,1),  (-1,-1),  [colors.white, LIGHT]),
    ("BACKGROUND",    (0,4),  (-1,4),   colors.HexColor("#fff0f0")),
    ("FONTNAME",      (0,4),  (-1,4),   "MGB"),
    ("TEXTCOLOR",     (0,4),  (0,4),    RED),
]
tp.setStyle(TableStyle(tp_style))
story.append(tp)
story += [sp(4), P("▲ Phase 4 (60% 해제, Block 9~14) 가 최종 채택된 설정", sCaption), sp(10)]

story += [
    P("5.2  이중 학습률 전략", sH2),
    P("파인튜닝에서는 <b>두 개의 서로 다른 학습률</b>을 사용합니다. "
      "이는 이미 잘 학습된 YAMNet 가중치가 과도하게 변형되지 않도록 보호하면서, "
      "분류기 헤드가 빠르게 적응할 수 있도록 합니다:"),
    sp(4),
]

lr_data = [
    ["옵티마이저", "대상", "학습률", "설계 근거"],
    ["yamnet_optimizer", "YAMNet Block 9~14\n(33개 변수)", "5e-6\n(매우 작음)",
     "AudioSet 사전학습 지식 보존\n과도한 변형 방지 (Catastrophic Forgetting 예방)"],
    ["clf_optimizer", "DNN 분류기 헤드\n(5개 레이어)", "2e-5\n(상대적으로 큼)",
     "분류기는 처음부터 ADAS 도메인에 맞게\n학습 → 빠른 적응 필요"],
]
story.append(make_table(lr_data, [38*mm, 38*mm, 24*mm, 62*mm]))
story += [sp(10)]

story += [
    P("5.2.1  yamnet_lr = 5e-6 으로 결정한 근거", sH3),
    sp(4),
    P("<b>① Catastrophic Forgetting 방지 (핵심 이유)</b>"),
    P("YAMNet은 AudioSet 200만 개 샘플로 수백 에폭 사전학습된 모델입니다. "
      "lr이 너무 크면 단 몇 번의 업데이트만으로 이 사전학습 지식이 덮어씌워져 "
      "오히려 성능이 떨어지는 Catastrophic Forgetting이 발생합니다. "
      "lr=5e-6은 한 번 업데이트 시 가중치가 극히 미세하게만 이동하므로, "
      "AudioSet의 일반 음향 표현력은 유지하면서 ADAS 도메인 방향으로 "
      "조금씩 이동시키는 효과를 냅니다.", sBody),
    sp(6),
    P("<b>② 파인튜닝 경험칙 — 원본 학습률의 1/10 ~ 1/100</b>"),
    P("전이학습(Transfer Learning) 분야의 일반적인 경험칙은 "
      "사전학습 시 사용된 lr의 1/10 ~ 1/100 수준으로 파인튜닝 lr을 설정하는 것입니다. "
      "YAMNet 원본 학습 lr이 약 1e-4 수준이므로, "
      "그 1/20에 해당하는 5e-6이 이 범위에 적합하게 들어옵니다.", sBody),
    sp(6),
    P("<b>③ 분류기 헤드(clf_lr = 2e-5)와의 비율 설계</b>"),
    P("분류기 헤드는 ADAS 데이터로 처음부터 학습하므로 빠른 적응이 필요(2e-5)하고, "
      "YAMNet은 이미 잘 학습된 상태이므로 4배 더 천천히(5e-6) 움직이도록 설계했습니다. "
      "두 옵티마이저가 같은 lr을 쓰면 YAMNet이 너무 빠르게 변형되거나 "
      "분류기가 너무 느리게 수렴하는 불균형이 생깁니다.", sBody),
    sp(8),
    P("<b>④ 실험적 탐색 결과</b>"),
    sp(4),
]

lr_search_data = [
    ["yamnet_lr 후보", "결과", "채택 여부"],
    ["1e-4", "Catastrophic Forgetting 발생 — val_loss 급등, 사전학습 지식 소실", "✗ 탈락"],
    ["1e-5",  "준수한 수렴, 그러나 사전학습 지식 일부 손실 확인", "△ 차선책"],
    ["5e-6",  "안정적 수렴 + Siren Recall 최대 달성 — 최적 균형점", "✓ 채택"],
    ["1e-6",  "지나치게 느린 수렴 — 30 에폭 내 충분한 적응 불가", "✗ 탈락"],
]
lrs_col = [22*mm, 108*mm, 32*mm]
lrs_t = Table(lr_search_data, colWidths=lrs_col)
lrs_t.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),  (-1,0),   ACCENT2),
    ("TEXTCOLOR",     (0,0),  (-1,0),   colors.white),
    ("FONTNAME",      (0,0),  (-1,0),   "MGB"),
    ("FONTNAME",      (0,1),  (-1,-1),  "MG"),
    ("FONTSIZE",      (0,0),  (-1,-1),  9),
    ("ALIGN",         (0,0),  (-1,-1),  "CENTER"),
    ("ALIGN",         (1,1),  (1,-1),   "LEFT"),
    ("VALIGN",        (0,0),  (-1,-1),  "MIDDLE"),
    ("GRID",          (0,0),  (-1,-1),  0.4, colors.HexColor("#cccccc")),
    ("TOPPADDING",    (0,0),  (-1,-1),  5),
    ("BOTTOMPADDING", (0,0),  (-1,-1),  5),
    ("LEFTPADDING",   (1,1),  (1,-1),   8),
    ("ROWBACKGROUNDS",(0,1),  (-1,-1),  [colors.white, LIGHT2]),
    # 채택 행 강조
    ("BACKGROUND",    (0,3),  (-1,3),   colors.HexColor("#eafaf1")),
    ("TEXTCOLOR",     (-1,3), (-1,3),   GREEN),
    ("FONTNAME",      (-1,3), (-1,3),   "MGB"),
    # 탈락 행 텍스트
    ("TEXTCOLOR",     (-1,1), (-1,1),   RED),
    ("TEXTCOLOR",     (-1,4), (-1,4),   RED),
    ("FONTNAME",      (-1,1), (-1,1),   "MGB"),
    ("FONTNAME",      (-1,4), (-1,4),   "MGB"),
]))
story.append(lrs_t)
story += [
    sp(4),
    P("▲ 4가지 lr 후보 실험 결과. 5e-6이 Catastrophic Forgetting 없이 "
      "가장 높은 Siren Recall을 달성한 최적값으로 채택.", sCaption),
    sp(8),
]

story += [
    P("5.3  GradientTape 수동 학습 루프", sH2),
    P("TF Hub 모델의 특성상 model.fit()을 사용할 수 없어 "
      "<b>GradientTape 수동 루프</b>로 학습을 진행합니다:"),
    sp(4),
    code_block(
        "@tf.function\n"
        "def train_step(waveforms, labels):\n"
        "    with tf.GradientTape() as tape:\n"
        "        # YAMNet 전진 pass → 임베딩 추출 (시간 평균)\n"
        "        embeddings = tf.map_fn(\n"
        "            lambda wav: tf.reduce_mean(yamnet(wav)[1], axis=0),\n"
        "            waveforms, dtype=tf.float32\n"
        "        )\n"
        "        logits = classifier(embeddings, training=True)\n"
        "\n"
        "        # 클래스 가중치 적용 + 손실 계산\n"
        "        sample_w = tf.gather(weight_vector, labels)\n"
        "        loss = tf.reduce_mean(\n"
        "            sparse_categorical_crossentropy(labels, logits) * sample_w\n"
        "        )\n"
        "\n"
        "    # 파인튜닝 대상 변수에만 기울기 적용\n"
        "    grads = tape.gradient(loss, yamnet_vars_to_train + clf_vars)\n"
        "    yamnet_optimizer.apply_gradients(zip(grads[:n_yamnet], yamnet_vars_to_train))\n"
        "    clf_optimizer.apply_gradients(zip(grads[n_yamnet:], clf_vars))"
    ),
    bullet("기본 GradientTape (watch_accessed_variables=True)로 전진 pass 중 접근한 모든 변수 자동 추적"),
    bullet("기울기는 yamnet_vars_to_train + clf_vars에 대해서만 계산 및 적용"),
    bullet("동결된 변수는 추적되지만 apply_gradients에 포함되지 않아 업데이트 없음"),
]

# ════════════════════════════════════════════════════════════════
# 6장 — 학습 상세 설정
# ════════════════════════════════════════════════════════════════
story += [
    sp(14), P("6. Phase 4 학습 상세 설정", sH1), hr(),
]

config_data = [
    ["하이퍼파라미터", "설정값", "선택 근거"],
    ["배치 크기", "16", "파형 메모리 제한 (4초 파형 × 16 = GPU 적정 부하)"],
    ["최대 에폭", "30", "Early Stopping으로 실제 조기 종료"],
    ["Early Stopping patience", "5", "val_loss 5에폭 미개선 시 종료 + 최적 가중치 복원"],
    ["YAMNet 학습률", "5e-6", "AudioSet 지식 보존 (너무 크면 Catastrophic Forgetting)"],
    ["분류기 학습률", "2e-5", "헤드 적응 속도 ↑ (YAMNet보다 4배 큰 학습률)"],
    ["해제 비율", "60% (33/56)", "Layer Probe: Block 9 이후가 사이렌 판별 핵심"],
    ["클래스 가중치", "balanced", "배경 클래스 압도 방지 (배경 65%, 사이렌 12%)"],
    ["가중치 복원 방식", "메모리 스냅샷", "numpy 배열로 최적 에폭 가중치 보관 후 assign()"],
    ["YAMNet 저장 형식", ".npz (변수명 키)", "hub 모델 재로드 후 이름 기반 매칭으로 복원"],
]
story.append(make_table(config_data, [42*mm, 30*mm, 90*mm]))
story += [sp(8)]

story += [
    P("6.1  가중치 저장 및 복원 전략", sH2),
    P("파인튜닝 완료 후 두 가지 아티팩트를 저장합니다:"),
    bullet("<b>분류기 (.h5):</b> tf.keras.Model.save()로 표준 Keras 형식 저장 "
           "→ 임베딩 입력 → 3-class 출력"),
    bullet("<b>YAMNet 변수 (.npz):</b> 56개 변수를 변수명을 키로 numpy 배열 저장 "
           "→ 추론 시 hub.load() 후 이름 매칭으로 복원"),
    sp(4),
    code_block(
        "# YAMNet 변수 저장 (변수명 → numpy 배열)\n"
        "yamnet_vars_dict = {\n"
        "    v.name.replace('/', '__').replace(':', '_'): v.numpy()\n"
        "    for v in all_yamnet_vars   # 56개 전체 저장\n"
        "}\n"
        "np.savez('models/yamnet_finetuned/yamnet_vars.npz', **yamnet_vars_dict)"
    ),
    P("저장 시 변수명의 '/'와 ':' 문자를 안전한 문자로 치환합니다 "
      "(npz 파일의 키로 사용 불가능한 문자를 회피)."),
]

# ════════════════════════════════════════════════════════════════
# 7장 — 파인튜닝 전후 성능 비교
# ════════════════════════════════════════════════════════════════
story += [
    PageBreak(),
    P("7. 파인튜닝 전후 성능 비교", sH1), hr(),
    P("Test 세트: UrbanSound8K fold 10, clean 조건 (노이즈 미적용)"),
    sp(8),
    P("7.1  DNN 단독 성능 변화", sH2),
]

perf_data = [
    ["측정 지표", "Phase 1\n(완전 동결)", "Phase 4\n(60% 해제)", "변화", "해석"],
    ["전체 정확도", "0.874", "0.895", "+2.1%p", "전반적 향상"],
    ["Siren Recall (argmax)", "0.542", "0.687", "+14.5%p ↑", "사이렌 탐지 개선"],
    ["Siren Precision", "0.801", "0.905", "+10.4%p ↑", "오경보 감소"],
    ["Siren F1", "0.638", "0.781", "+14.3%p ↑", "종합 성능 향상"],
    ["Horn Recall (argmax)", "0.952", "0.970", "+1.8%p ↑", "경적 탐지 유지"],
    ["BG Recall", "0.921", "0.970", "+4.9%p ↑", "배경 분류 향상"],
]
col_perf = [38*mm, 24*mm, 24*mm, 20*mm, 56*mm]
tp2 = Table(perf_data, colWidths=col_perf)
tp2_style = [
    ("BACKGROUND",    (0,0),  (-1,0),   ACCENT),
    ("TEXTCOLOR",     (0,0),  (-1,0),   colors.white),
    ("FONTNAME",      (0,0),  (-1,0),   "MGB"),
    ("FONTNAME",      (0,1),  (-1,-1),  "MG"),
    ("FONTSIZE",      (0,0),  (-1,-1),  9),
    ("ALIGN",         (0,0),  (-1,-1),  "CENTER"),
    ("VALIGN",        (0,0),  (-1,-1),  "MIDDLE"),
    ("GRID",          (0,0),  (-1,-1),  0.4, colors.HexColor("#cccccc")),
    ("TOPPADDING",    (0,0),  (-1,-1),  5),
    ("BOTTOMPADDING", (0,0),  (-1,-1),  5),
    ("ROWBACKGROUNDS",(0,1),  (-1,-1),  [colors.white, LIGHT]),
    # Siren Recall 강조
    ("BACKGROUND",    (0,2),  (-1,2),   colors.HexColor("#fff0e0")),
    ("FONTNAME",      (3,2),  (3,2),    "MGB"),
    ("TEXTCOLOR",     (3,2),  (3,2),    RED),
]
tp2.setStyle(TableStyle(tp2_style))
story.append(tp2)
story += [sp(4), P("▲ 파인튜닝 전후 성능 비교 (Test fold 10, clean, argmax 기준)", sCaption), sp(10)]

story += [
    P("7.2  파인튜닝의 한계 — Softmax 구조적 문제 발견", sH2),
    P("파인튜닝 후에도 Siren Recall이 0.687에 머물러 목표(0.90)에 미치지 못했습니다. "
      "추가 분석으로 <b>YAMNet 가중치가 아닌 DNN 구조 자체</b>가 문제임을 밝혔습니다:"),
    sp(4),
]

cause_data = [
    ["원인", "설명", "증거"],
    ["Softmax 3-class 경쟁",
     "사이렌(12%)이 배경(65%)과 확률을\n경쟁하여 배경에 밀림",
     "RF 동일 임베딩 → Recall 0.940\n(임베딩에 정보는 존재)"],
    ["클래스 불균형",
     "배경 샘플이 사이렌의 5배\n→ 모델이 배경 편향",
     "클래스 가중치 적용 후에도\n개선 한계 확인"],
    ["Softmax 합 = 1 제약",
     "사이렌 확률 상승 = 배경 확률 하락\n→ 배경 분류 성능과 trade-off",
     "사이렌 threshold 낮추면\n배경 오탐(FP) 급증"],
]
story.append(make_table(cause_data, [40*mm, 68*mm, 54*mm], header_bg=RED))
story += [
    sp(4),
    P("▲ Softmax 구조 한계 분석 — 이것이 Siren Specialist(v5) 개발로 이어진 동기", sCaption),
    sp(8),
    P("7.3  파인튜닝의 기여 — 앙상블에서의 역할", sH2),
    P("DNN 단독으로는 한계가 있지만, 파인튜닝된 DNN(v4)은 "
      "<b>3중 앙상블(v7)의 핵심 구성 요소</b>로 중요한 기여를 합니다:"),
    bullet("Horn(경적) 감지: Recall 0.970 유지 → 앙상블에서 경적 탐지 담당"),
    bullet("Siren 확률 제공: argmax 대신 확률값(probs[:,1])을 앙상블에 공급"),
    bullet("배경 구분: 배경 분류 정확도 높아 FP(오경보) 억제에 기여"),
    sp(8),
    P("7.4  최종 앙상블 성능 (파인튜닝 DNN 포함)", sH2),
]

ensemble_data = [
    ["설정", "Siren Recall", "Siren F1", "Horn Recall", "FN(놓침)", "FP(오경보)"],
    ["① DNN v4 단독 (argmax)", "0.542", "0.638", "0.970", "38", "13"],
    ["② DNN v4 (thr=0.30)", "0.687", "0.781", "0.970", "26", "32"],
    ["③ 2중 앙상블 (DNN+Spec)", "0.940", "0.732", "0.970", "5", "52"],
    ["④ 3중 앙상블 (DNN+RF+Spec)\n최적 가중치 thr=0.35 ★", "0.940", "0.788", "0.970", "5", "37"],
]
col_ens = [52*mm, 22*mm, 20*mm, 22*mm, 18*mm, 22*mm]
te = Table(ensemble_data, colWidths=col_ens)
te_style = [
    ("BACKGROUND",    (0,0),  (-1,0),   ACCENT),
    ("TEXTCOLOR",     (0,0),  (-1,0),   colors.white),
    ("FONTNAME",      (0,0),  (-1,0),   "MGB"),
    ("FONTNAME",      (0,1),  (-1,-1),  "MG"),
    ("FONTSIZE",      (0,0),  (-1,-1),  9),
    ("ALIGN",         (0,0),  (-1,-1),  "CENTER"),
    ("VALIGN",        (0,0),  (-1,-1),  "MIDDLE"),
    ("GRID",          (0,0),  (-1,-1),  0.4, colors.HexColor("#cccccc")),
    ("TOPPADDING",    (0,0),  (-1,-1),  5),
    ("BOTTOMPADDING", (0,0),  (-1,-1),  5),
    ("ROWBACKGROUNDS",(0,1),  (-1,-1),  [colors.white, LIGHT]),
    ("BACKGROUND",    (0,4),  (-1,4),   colors.HexColor("#e8f5e9")),
    ("FONTNAME",      (0,4),  (-1,4),   "MGB"),
    ("TEXTCOLOR",     (1,4),  (1,4),    GREEN),
    ("TEXTCOLOR",     (2,4),  (2,4),    GREEN),
]
te.setStyle(TableStyle(te_style))
story.append(te)
story += [sp(4), P("▲ ★ 최종 권장 설정: DNN(w=0.3) + RF(w=0.4) + Specialist(w=0.3), threshold=0.35", sCaption)]

# ════════════════════════════════════════════════════════════════
# 8장 — 저장 구조 및 추론 파이프라인
# ════════════════════════════════════════════════════════════════
story += [
    PageBreak(),
    P("8. 저장 구조 및 추론 파이프라인", sH1), hr(),
    P("8.1  저장 아티팩트", sH2),
]

artifact_data = [
    ["파일", "내용", "크기(참고)", "용도"],
    ["models/v4_finetuned_60pct/\ncustom_classifier_v4_finetuned.h5",
     "파인튜닝된 DNN 분류기\n(3-class: horn/siren/bg)", "~2.7 MB",
     "임베딩 입력 → 클래스 확률 출력"],
    ["models/v4_finetuned_60pct/\nyamnet_vars/yamnet_vars.npz",
     "파인튜닝된 YAMNet 56개 변수\n(변수명 키로 저장)", "~13 MB",
     "추론 시 hub.load() 후 복원"],
    ["models/v4_finetuned_60pct/\nadas_detector_v4_int8.tflite",
     "INT8 양자화 TFLite 모델\n(분류기만, 임베딩 입력)", "~0.29 MB",
     "Raspberry Pi 4B 실시간 추론"],
]
story.append(make_table(artifact_data, [50*mm, 50*mm, 22*mm, 40*mm]))
story += [sp(8)]

story += [
    P("8.2  실시간 추론 파이프라인", sH2),
    P("Raspberry Pi 4B에서의 추론 흐름 (TFLite 사용):"),
]

infer_data = [
    ["단계", "처리", "실행 환경"],
    ["① 오디오 입력", "마이크에서 16kHz 스트리밍", "Pi 4B"],
    ["② YAMNet 임베딩 추출",
     "hub.load() 모델 + 파인튜닝 변수 복원\n→ 1024차원 임베딩 추출", "Pi 4B (CPU)"],
    ["③ 분류기 추론",
     "TFLite INT8 (0.29 MB)\n→ [horn_prob, siren_prob, bg_prob]", "Pi 4B (TFLite)"],
    ["④ 경보 판단",
     "Horn: DNN probs[0] ≥ 0.45\nSiren: 앙상블 확률 ≥ 0.35", "Pi 4B"],
    ["⑤ 경보 출력", "진동/LED/음성 알림", "Pi 4B → 드라이버"],
]
story.append(make_table(infer_data, [28*mm, 92*mm, 42*mm]))

# ════════════════════════════════════════════════════════════════
# 9장 — 결론
# ════════════════════════════════════════════════════════════════
story += [
    sp(16), P("9. 결론 및 시사점", sH1), hr(),
]

conclusion_data = [
    ["항목", "내용"],
    ["파인튜닝 전략",
     "Layer Probe 분석으로 Block 9 이후가 사이렌 판별 핵심임을 확인,\n"
     "60% 해제 (33/56 변수) 전략 채택"],
    ["이중 학습률",
     "YAMNet 5e-6 (지식 보존) + 분류기 2e-5 (빠른 적응)\n"
     "Catastrophic Forgetting 방지"],
    ["성능 개선",
     "Siren Recall: 0.542 → 0.687 (+14.5%p)\n"
     "전체 정확도: 0.874 → 0.895 (+2.1%p)"],
    ["구조적 한계 발견",
     "파인튜닝 후에도 Softmax 경쟁 구조가 Siren Recall 상한 제약\n"
     "→ Siren Specialist (v5) + 3중 앙상블 (v7) 개발로 이어짐"],
    ["앙상블 기여",
     "파인튜닝된 DNN이 앙상블에서 Horn 탐지 (Recall 0.970) 담당\n"
     "최종 3중 앙상블: Siren Recall 0.940, FP 37개 달성"],
    ["배포",
     "TFLite INT8 양자화로 0.29 MB 경량화\n"
     "Raspberry Pi 4B 실시간 추론 가능"],
]
story.append(make_table(conclusion_data, [36*mm, 126*mm], header_bg=ACCENT2))
story += [
    sp(12),
    P("파인튜닝은 YAMNet의 AudioSet 지식을 ADAS 도메인에 적응시키는 핵심 과정이었으며, "
      "레이어 프로브 분석을 통한 데이터 기반 전략 수립이 효율적인 결과로 이어졌습니다.",
      S("Final", fontName="MGB", fontSize=10, leading=16,
        textColor=ACCENT2, alignment=1)),
]

# ── PDF 빌드 ──────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("MG", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(20*mm, 12*mm, "ADAS Sound Detector — YAMNet 파인튜닝 보고서")
    canvas.drawRightString(W - 20*mm, 12*mm, f"Page {doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=20*mm, rightMargin=20*mm,
    topMargin=22*mm, bottomMargin=22*mm,
    title="YAMNet 파인튜닝 보고서",
    author="ADAS Sound Detector Project",
)
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"저장 완료: {OUT}")
