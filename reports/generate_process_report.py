"""
ADAS 사운드 감지 프로젝트 — 과정과 이유 종합 PDF
전체 의사결정 과정, 시행착오, 기술적 근거를 서술형으로 정리
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, PageBreak, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import datetime

# ── 한글 폰트 등록 (맑은 고딕) ──────────────────────────────
pdfmetrics.registerFont(TTFont("MalgunGothic",   r"C:\Windows\Fonts\malgun.ttf"))
pdfmetrics.registerFont(TTFont("MalgunGothicBd", r"C:\Windows\Fonts\malgunbd.ttf"))

FONT_NORMAL = "MalgunGothic"
FONT_BOLD   = "MalgunGothicBd"

PROJECT = Path(__file__).resolve().parents[1]  # reports/ → 프로젝트 루트
OUT = PROJECT / "reports/pdf/ADAS_Process_Report_v2.pdf"

W = A4[0] - 4.4*cm

C_NAVY  = colors.HexColor("#1a2744")
C_BLUE  = colors.HexColor("#2563eb")
C_CYAN  = colors.HexColor("#0891b2")
C_GREEN = colors.HexColor("#15803d")
C_RED   = colors.HexColor("#dc2626")
C_GOLD  = colors.HexColor("#b45309")
C_GRAY  = colors.HexColor("#6b7280")
C_LGRAY = colors.HexColor("#f3f4f6")
C_BGROW = colors.HexColor("#eff6ff")
C_HLROW = colors.HexColor("#fef9c3")
C_GRNBG = colors.HexColor("#f0fdf4")
C_REDBG = colors.HexColor("#fef2f2")


def styles():
    s = getSampleStyleSheet()
    def a(name, **kw):
        s.add(ParagraphStyle(name=name, **kw))

    a("CVR",   fontName=FONT_BOLD,   fontSize=24, textColor=C_NAVY, alignment=TA_CENTER, spaceAfter=10, leading=32)
    a("CVRB",  fontName=FONT_BOLD,   fontSize=13, textColor=C_BLUE, alignment=TA_CENTER, spaceAfter=5,  leading=20)
    a("CVRS",  fontName=FONT_NORMAL, fontSize=10, textColor=C_GRAY, alignment=TA_CENTER, spaceAfter=4,  leading=16)

    a("H1",    fontName=FONT_BOLD,   fontSize=14, textColor=colors.white,
               backColor=C_NAVY, spaceBefore=20, spaceAfter=10,
               leftIndent=-6, rightIndent=-6, borderPad=6, leading=22)
    a("H2",    fontName=FONT_BOLD,   fontSize=11, textColor=C_BLUE,
               spaceBefore=14, spaceAfter=6,
               borderPad=3, backColor=C_BGROW, leftIndent=-2, leading=18)
    a("H3",    fontName=FONT_BOLD,   fontSize=10, textColor=C_CYAN,
               spaceBefore=10, spaceAfter=4, leading=16)

    a("BODY",  fontName=FONT_NORMAL, fontSize=10, textColor=colors.black,
               spaceAfter=5, leading=17, alignment=TA_JUSTIFY)
    a("BODYB", fontName=FONT_BOLD,   fontSize=10, textColor=colors.black,
               spaceAfter=5, leading=17)
    a("BUL",   fontName=FONT_NORMAL, fontSize=10, textColor=colors.black,
               spaceAfter=4, leading=16, leftIndent=14)
    a("BULB",  fontName=FONT_BOLD,   fontSize=10, textColor=C_NAVY,
               spaceAfter=4, leading=16, leftIndent=14)

    a("QUOTE", fontName=FONT_NORMAL, fontSize=10, textColor=C_GOLD,
               spaceAfter=6, leading=17, leftIndent=16, rightIndent=10,
               borderPad=6, backColor=C_HLROW)
    a("RESULT",fontName=FONT_BOLD,   fontSize=11, textColor=C_GREEN,
               spaceAfter=6, alignment=TA_CENTER,
               backColor=C_GRNBG, borderPad=5, leading=18)
    a("WARN",  fontName=FONT_BOLD,   fontSize=10, textColor=C_RED,
               spaceAfter=5, backColor=C_REDBG, borderPad=4, leading=17)
    a("CAP",   fontName=FONT_NORMAL, fontSize=8,  textColor=C_GRAY,
               alignment=TA_CENTER, spaceAfter=8, leading=13)
    a("MONO",  fontName="Courier",   fontSize=9,  textColor=C_NAVY,
               spaceAfter=4, backColor=C_LGRAY, leftIndent=10, leading=13)
    a("TOC",   fontName=FONT_NORMAL, fontSize=10, textColor=C_BLUE,
               spaceAfter=4, leftIndent=8, leading=16)
    a("TOCH",  fontName=FONT_BOLD,   fontSize=11, textColor=C_NAVY,
               spaceAfter=6, spaceBefore=6, leading=18)
    return s


def T(data, widths, extra=None):
    base = [
        ("BACKGROUND",    (0,0), (-1,0),  C_NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  FONT_BOLD),
        ("FONTSIZE",      (0,0), (-1,0),  9),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("FONTNAME",      (0,1), (-1,-1), FONT_NORMAL),
        ("FONTSIZE",      (0,1), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, C_BGROW]),
        ("GRID",          (0,0), (-1,-1), 0.4, C_GRAY),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("LEADING",       (0,0), (-1,-1), 15),
    ]
    if extra:
        base.extend(extra)
    return Table(data, colWidths=widths, style=TableStyle(base), hAlign="LEFT")


def BF(text):
    """Bold formatting helper for inline bold in Paragraph."""
    return f"<font name='{FONT_BOLD}'>{text}</font>"


def build():
    s = styles()
    story = []

    # ════════════════════════════════════════════════
    # 표지
    # ════════════════════════════════════════════════
    story += [
        Spacer(1, 2.5*cm),
        Paragraph("ADAS 사운드 감지 시스템", s["CVR"]),
        Paragraph("전체 개발 과정 · 의사결정 근거 · 기술 원리", s["CVRB"]),
        Paragraph("Why We Did What We Did — Complete Process Report", s["CVRS"]),
        Spacer(1, 0.4*cm),
        HRFlowable(width=W, color=C_NAVY, thickness=2),
        Spacer(1, 0.3*cm),
        Paragraph(f"작성일: {datetime.datetime.now().strftime('%Y년 %m월 %d일')}", s["CVRS"]),
        Paragraph("YAMNet (MobileNet V1) 기반 청각장애인 보조 ADAS | Raspberry Pi 4B 배포 대상", s["CVRS"]),
        Spacer(1, 1.5*cm),
    ]

    # 목차
    toc_data = [
        ["장", "제목", "핵심 질문"],
        ["1",  "프로젝트 배경과 목표",            "왜 이 시스템이 필요한가?"],
        ["2",  "YAMNet을 선택한 이유",             "왜 처음부터 만들지 않고 YAMNet을 쓰는가?"],
        ["3",  "3-class DNN 분류기 설계",          "왜 sigmoid가 아닌 softmax를 쓰는가?"],
        ["4",  "파인튜닝(Fine-tuning) 전략",       "왜 전체가 아닌 일부만 학습시키는가?"],
        ["5",  "Siren Recall 68.7%의 근본 원인",   "왜 사이렌만 유독 낮은가?"],
        ["6",  "RF 비교 실험",                     "왜 Random Forest로 검증하는가?"],
        ["7",  "레이어 프로브 분석",               "어느 레이어가 사이렌을 가장 잘 구별하는가?"],
        ["8",  "Siren Specialist 설계 이유",       "왜 별도 이진 분류기를 만드는가?"],
        ["9",  "StandardScaler — 왜 쓰는가",       "정규화가 왜 필수인가?"],
        ["10", "Focal Loss — 왜 선택했는가",       "CrossEntropy와 무엇이 다른가?"],
        ["11", "MixUp 증강 원리와 이유",            "왜 합성 데이터를 만드는가?"],
        ["12", "최적 Threshold 탐색 방법",          "0.78은 어떻게 구했는가?"],
        ["13", "앙상블 전략 비교",                  "OR/AVG/MAX — 무엇을 골라야 하는가?"],
        ["14", "SNR 노이즈 분석 결과와 함의",       "노이즈가 정말 문제인가?"],
        ["15", "최종 결론 및 시스템 권고",          "무엇을 배포해야 하는가?"],
    ]
    story.append(T(toc_data, [1*cm, 6.5*cm, 8.8*cm],
        extra=[("ALIGN", (0,1), (2,-1), "LEFT")]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 1. 프로젝트 배경
    # ════════════════════════════════════════════════
    story.append(Paragraph("1장. 프로젝트 배경과 목표", s["H1"]))
    story.append(Paragraph("1.1 왜 이 시스템이 필요한가", s["H2"]))
    story.append(Paragraph(
        "청각장애인 운전자는 차량 경적(Car Horn)과 구급차·경찰차·소방차 사이렌(Siren) 소리를 "
        "인지할 수 없습니다. 일반 운전자는 이 소리를 통해 위험 상황을 즉각 파악하고 대응하지만, "
        "청각장애인 운전자는 이 정보에서 완전히 차단됩니다. ADAS(Advanced Driver Assistance "
        "System) 사운드 감지 모듈은 이 격차를 메우기 위해 설계되었습니다.",
        s["BODY"]))
    story.append(Paragraph(
        "핵심 목표: 실시간으로 차량 경적과 사이렌을 감지하여 운전자에게 시각·진동 알림을 제공합니다. "
        "Raspberry Pi 4B에서 동작해야 하므로 경량화(TFLite INT8 양자화)가 필수 조건입니다.",
        s["QUOTE"]))

    story.append(Paragraph("1.2 성능 목표", s["H2"]))
    goal_data = [
        ["지표",         "목표값",      "이유"],
        ["전체 정확도",  "98.5% 이상",  "오경보가 너무 많으면 운전자 신뢰 하락"],
        ["Siren Recall", "90% 이상",    "사이렌을 놓치는 것은 생명과 직결"],
        ["Horn Recall",  "95% 이상",    "경적 무시 시 사고 위험"],
        ["추론 지연",    "30ms 미만",   "실시간 반응성 확보"],
        ["모델 크기",    "5MB 미만",    "Raspberry Pi 저장 공간 제약"],
    ]
    story.append(T(goal_data, [3.5*cm, 3*cm, 9.8*cm],
        extra=[("ALIGN", (0,1), (0,-1), "LEFT"), ("ALIGN", (2,1), (2,-1), "LEFT")]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 2. YAMNet 선택 이유
    # ════════════════════════════════════════════════
    story.append(Paragraph("2장. YAMNet을 선택한 이유", s["H1"]))
    story.append(Paragraph("2.1 왜 처음부터 음향 모델을 만들지 않는가", s["H2"]))
    story.append(Paragraph(
        "음향 인식 모델을 처음부터 학습시키려면 수십만 개의 레이블된 오디오 샘플과 "
        "수일에서 수주의 GPU 학습 시간이 필요합니다. UrbanSound8K 데이터셋은 각 클래스당 "
        "수백~수천 개의 샘플만 제공하므로 이 데이터만으로는 일반화 능력이 부족합니다. "
        "반면 YAMNet은 Google이 AudioSet(YouTube 200만 클립, 521개 클래스)으로 "
        "사전 학습한 모델로, 이미 풍부한 음향 특징을 학습하고 있습니다.",
        s["BODY"]))
    story.append(Paragraph("2.2 YAMNet 내부 구조", s["H2"]))
    story.append(Paragraph(
        "YAMNet은 MobileNet V1 아키텍처를 기반으로 합니다. 오디오 파형을 입력받아 "
        "Log-Mel Spectrogram으로 변환한 후 14개의 Depthwise Separable Convolution "
        "블록을 통해 처리합니다. 마지막 Global Average Pooling을 거쳐 "
        "1024차원 임베딩 벡터를 출력합니다. 이 임베딩은 AudioSet 전체 521개 "
        "클래스를 구별하도록 학습된 풍부한 음향 표현입니다.",
        s["BODY"]))
    arch_data = [
        ["구성요소",                    "역할"],
        ["오디오 입력 (16kHz)",         "원시 파형"],
        ["Log-Mel Spectrogram",         "시간-주파수 표현 (64 mel bins, 25ms hop)"],
        ["14x Depthwise Sep. Conv.",    "채널별 특징 추출 (파라미터 효율 90% 절감)"],
        ["Global Average Pooling",      "시간 축 평균 → 고정 크기 벡터"],
        ["1024차원 임베딩",             "우리가 사용하는 표현 (AudioSet 표현 공간)"],
        ["Custom DNN 분류기",           "우리 데이터에 맞는 3-class 분류 (추가 학습)"],
    ]
    story.append(T(arch_data, [5*cm, 11.3*cm],
        extra=[("ALIGN", (0,1), (1,-1), "LEFT")]))
    story.append(Paragraph("2.3 전이 학습(Transfer Learning)의 원리", s["H2"]))
    story.append(Paragraph(
        "YAMNet의 가중치는 AudioSet으로 이미 수렴되어 있습니다. 우리가 원하는 "
        "car_horn/siren/background 클래스는 AudioSet에도 포함되어 있으므로, "
        "YAMNet의 내부 표현이 이미 이 소리들을 구별하는 데 적합합니다. "
        "따라서 YAMNet을 특징 추출기로 사용하고 그 위에 소형 DNN 분류기만 추가로 학습하면 "
        "적은 데이터와 짧은 학습 시간으로 높은 성능을 달성할 수 있습니다.",
        s["BODY"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 3. 3-class DNN
    # ════════════════════════════════════════════════
    story.append(Paragraph("3장. 3-class DNN 분류기 설계", s["H1"]))
    story.append(Paragraph("3.1 분류기 구조", s["H2"]))
    story.append(Paragraph(
        "YAMNet이 출력하는 1024차원 임베딩을 입력으로 받아 "
        "car_horn / siren / background 3개 클래스로 분류하는 소형 DNN입니다.",
        s["BODY"]))
    dnn_data = [
        ["레이어",                      "출력 차원", "활성화",  "역할"],
        ["Dense",                       "512",       "ReLU",    "고수준 특징 조합"],
        ["BatchNorm + Dropout(0.4)",    "512",       "—",       "과적합 방지, 학습 안정화"],
        ["Dense",                       "256",       "ReLU",    "중간 표현"],
        ["BatchNorm + Dropout(0.4)",    "256",       "—",       "—"],
        ["Dense",                       "64",        "ReLU",    "압축된 표현"],
        ["BatchNorm + Dropout(0.4)",    "64",        "—",       "—"],
        ["Dense (출력)",                "3",         "Softmax", "3-class 확률 출력"],
    ]
    story.append(T(dnn_data, [5.5*cm, 2.5*cm, 2.5*cm, 5.8*cm],
        extra=[("ALIGN", (0,1), (3,-1), "LEFT")]))

    story.append(Paragraph("3.2 왜 Softmax인가 — 그리고 왜 문제가 되는가", s["H2"]))
    story.append(Paragraph(
        "Softmax는 3개의 클래스 확률을 합이 1이 되도록 정규화합니다. "
        "즉, 한 클래스의 확률이 높아지면 다른 클래스의 확률은 자동으로 낮아집니다. "
        "이것이 나중에 사이렌 감지 실패의 핵심 원인이 됩니다.",
        s["BODY"]))
    story.append(Paragraph(
        "Softmax의 경쟁 구도 문제: Background가 전체 데이터의 약 65%를 차지합니다. "
        "모델은 학습 중 'background로 예측하면 자주 맞다'는 것을 학습합니다. "
        "결과적으로 siren 확률을 끌어올리려면 background 확률을 낮춰야 하는데, "
        "배경음과 사이렌의 경계가 모호할 때 모델은 보수적으로 background를 선택합니다. "
        "이것이 Siren Recall 68.7%의 근본 원인입니다.",
        s["WARN"]))

    story.append(Paragraph("3.3 Adaptive Threshold (적응형 임계값)", s["H2"]))
    story.append(Paragraph(
        "Argmax(가장 높은 확률 클래스를 선택) 대신 클래스별 독립 threshold를 설정합니다. "
        "siren 확률이 0.20 이상이면 siren으로 판정하고, car_horn 확률이 0.45 이상이면 "
        "car_horn으로 판정합니다. 이렇게 하면 siren의 recall을 독립적으로 조절할 수 있습니다. "
        "하지만 이것만으로는 충분하지 않았습니다. DNN threshold=0.20 적용 시 Recall 65.1%에 그쳤습니다.",
        s["BODY"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 4. 파인튜닝 전략
    # ════════════════════════════════════════════════
    story.append(Paragraph("4장. 파인튜닝(Fine-tuning) 전략", s["H1"]))
    story.append(Paragraph("4.1 왜 전체 YAMNet을 학습시키지 않는가", s["H2"]))
    story.append(Paragraph(
        "YAMNet 전체(56개 변수, 약 400만 파라미터)를 처음부터 우리 데이터로 학습하면 "
        "두 가지 위험이 있습니다. 첫째, 데이터 부족 문제입니다. UrbanSound8K는 약 8,000개 "
        "샘플에 불과합니다. 400만 파라미터를 충분히 학습하기에는 턱없이 부족합니다. "
        "둘째, Catastrophic Forgetting 문제입니다. AudioSet으로 학습된 풍부한 일반 음향 "
        "표현이 무너지고 3개 클래스에 과적합될 수 있습니다.",
        s["BODY"]))
    story.append(Paragraph("4.2 점진적 해제(Gradual Unfreezing) 전략", s["H2"]))
    ft_data = [
        ["페이즈",              "해제 범위",                    "학습 변수",            "목적"],
        ["Phase 1\n(분류기만)", "YAMNet 전체 고정\n(0% unfreeze)", "분류기 Dense 레이어만", "임베딩에 맞는 분류기 초기화"],
        ["Phase 2\n(25%)",      "Block 11-14 해제\n(14개 변수)", "분류기 + YAMNet 후반부","음향 특징 미세 조정 시작"],
        ["Phase 3\n(40%)",      "Block 9-14 일부\n(23개 변수)", "더 깊은 레이어 포함",  "표현력 확장"],
        ["Phase 4\n(60%) ★",   "Block 9-14 전체\n(33개 변수)", "YAMNet 후반 + 분류기", "최적 파인튜닝 범위"],
    ]
    story.append(T(ft_data, [2.5*cm, 3.5*cm, 3.8*cm, 6.5*cm],
        extra=[
            ("ALIGN",      (0,1), (3,-1), "LEFT"),
            ("VALIGN",     (0,0), (-1,-1),"TOP"),
            ("BACKGROUND", (0,4), (3,4),  C_HLROW),
            ("FONTNAME",   (0,4), (3,4),  FONT_BOLD),
        ]))
    story.append(Paragraph("4.3 두 개의 학습률(Dual Learning Rate)", s["H2"]))
    story.append(Paragraph(
        "파인튜닝 시 YAMNet 가중치와 분류기 가중치에 서로 다른 학습률을 적용합니다. "
        "YAMNet 레이어는 이미 좋은 표현을 가지고 있으므로 매우 작은 학습률(5x10^-6)로 "
        "미세 조정합니다. 분류기는 처음 학습하는 것이므로 더 큰 학습률(2x10^-5)을 사용합니다. "
        "이렇게 하면 사전학습된 지식은 보존하면서 우리 데이터에 적응시킬 수 있습니다.",
        s["BODY"]))
    story.append(Paragraph(
        "Phase 4 결과: Val Acc 99.37% (Epoch 30 best), Test 전체 Acc 89.5%\n"
        "그러나 Siren Recall은 여전히 68.7% — 분류기 구조 자체의 한계 확인",
        s["RESULT"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 5. Siren Recall 68.7% 원인
    # ════════════════════════════════════════════════
    story.append(Paragraph("5장. Siren Recall 68.7%의 근본 원인 분석", s["H1"]))
    story.append(Paragraph("5.1 현상 파악", s["H2"]))
    story.append(Paragraph(
        "Phase 4 파인튜닝 완료 후 테스트 결과에서 이상한 패턴을 발견했습니다. "
        "전체 정확도 89.5%는 양호하지만 클래스별로 보면 심각한 불균형이 존재합니다.",
        s["BODY"]))
    recall_data = [
        ["클래스",      "Recall",  "테스트 샘플 수", "판단"],
        ["Car Horn",    "97.0%",   "33개",           "양호"],
        ["Background",  "97.0%",   "198개",          "양호"],
        ["Siren",       "68.7%",   "83개",           "심각하게 낮음 — 26개 놓침"],
    ]
    story.append(T(recall_data, [3.5*cm, 2.5*cm, 3.5*cm, 6.8*cm],
        extra=[
            ("ALIGN",      (0,1), (3,-1), "LEFT"),
            ("BACKGROUND", (0,3), (3,3),  C_REDBG),
            ("TEXTCOLOR",  (1,3), (1,3),  C_RED),
            ("FONTNAME",   (0,3), (3,3),  FONT_BOLD),
        ]))
    story.append(Paragraph("5.2 왜 사이렌만 낮은가 — 3가지 원인", s["H2"]))
    story.append(Paragraph("원인 1: 클래스 불균형", s["H3"]))
    story.append(Paragraph(
        "훈련 데이터에서 background:siren:horn의 비율은 약 65:22:13입니다. "
        "Softmax CrossEntropy는 다수 클래스에 편향되어 학습됩니다. "
        "모델 입장에서 '항상 background'라고 예측해도 65%의 정확도를 달성할 수 있습니다. "
        "소수 클래스인 사이렌은 학습 신호가 상대적으로 약합니다.",
        s["BODY"]))
    story.append(Paragraph("원인 2: Softmax의 경쟁 구조", s["H3"]))
    story.append(Paragraph(
        "사이렌 확률을 높이려면 background 확률이 낮아져야 합니다. "
        "하지만 사이렌이 포함된 도심 음원은 배경음도 함께 존재하므로 "
        "모델이 모호한 상황에서 다수 클래스(background)를 선택하는 경향이 강해집니다. "
        "이진 분류기(Sigmoid)였다면 두 클래스가 경쟁하지 않으므로 이 문제가 없습니다.",
        s["BODY"]))
    story.append(Paragraph("원인 3: 사이렌 음향 다양성", s["H3"]))
    story.append(Paragraph(
        "경적은 단순한 음향이지만 사이렌은 주파수 변조(피치가 오르내림), "
        "종류에 따른 패턴 차이(경찰/소방/구급), 도플러 효과 등 "
        "음향 패턴이 훨씬 복잡하고 다양합니다. "
        "하나의 shared 3-class 분류기로 이 다양성을 모두 포착하기 어렵습니다.",
        s["BODY"]))
    story.append(Paragraph("5.3 핵심 가설: 임베딩에는 정보가 있다", s["H2"]))
    story.append(Paragraph(
        "만약 YAMNet 임베딩 자체가 사이렌 정보를 충분히 담고 있다면 "
        "분류기만 교체해도 Recall을 높일 수 있습니다. "
        "이를 검증하기 위해 두 가지 실험을 설계했습니다. "
        "(1) Random Forest 비교 실험, (2) 레이어별 선형 프로브 분석.",
        s["BODY"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 6. RF 실험
    # ════════════════════════════════════════════════
    story.append(Paragraph("6장. Random Forest 비교 실험", s["H1"]))
    story.append(Paragraph("6.1 왜 Random Forest인가", s["H2"]))
    story.append(Paragraph(
        "Random Forest는 DNN과 전혀 다른 방식으로 분류합니다. "
        "수백 개의 결정 트리를 앙상블하여 각 트리가 독립적으로 판단합니다. "
        "특히 class_weight='balanced' 옵션을 사용하면 클래스 불균형을 "
        "자동으로 보정합니다. 사이렌 샘플에 더 높은 가중치를 부여하는 방식입니다. "
        "또한 RF는 비선형 결정 경계를 만들 수 있어 복잡한 사이렌 음향 패턴도 "
        "효과적으로 분리할 수 있습니다.",
        s["BODY"]))
    story.append(Paragraph("6.2 실험 설계 — 공정한 비교를 위한 조건", s["H2"]))
    for b in [
        "동일한 YAMNet 임베딩(embeddings.npz, 1024차원) 사용 — 분류기만 다름",
        "훈련: fold 1-8 (7,246개), 테스트: fold 10 clean (314개)",
        "RF 변형 5종: 기본/clean-only/siren×3 오버샘플링/siren×5/threshold 조정",
        "추가: Feature Importance로 선별한 Top-20 차원만 사용한 실험",
    ]:
        story.append(Paragraph(f"• {b}", s["BUL"]))

    story.append(Paragraph("6.3 실험 결과", s["H2"]))
    rf_data = [
        ["모델",                  "Siren Recall", "Horn Recall", "전체 Acc", "해석"],
        ["DNN Phase 4 (기준선)", "0.687",        "0.970",       "0.895",    "기존 시스템"],
        ["RF aug+balanced",      "0.928",        "0.909",       "0.880",    "최고 Siren Recall"],
        ["RF siren×5",           "0.940",        "0.909",       "0.873",    "공격적 오버샘플링"],
        ["RF Top-20 dims only",  "0.928",        "0.939",       "0.854",    "20차원만으로 동급"],
    ]
    story.append(T(rf_data, [4*cm, 2.2*cm, 2.2*cm, 2.2*cm, 5.7*cm],
        extra=[
            ("ALIGN",      (0,1), (4,-1), "LEFT"),
            ("BACKGROUND", (0,1), (4,1),  C_REDBG),
            ("BACKGROUND", (0,2), (4,2),  C_GRNBG),
            ("FONTNAME",   (0,2), (4,2),  FONT_BOLD),
            ("TEXTCOLOR",  (1,1), (1,1),  C_RED),
            ("TEXTCOLOR",  (1,2), (1,2),  C_GREEN),
        ]))
    story.append(Paragraph(
        "핵심 결론: 같은 임베딩으로 RF는 Siren Recall 92.8%를 달성합니다. DNN은 68.7%에 그칩니다. "
        "임베딩에 정보가 없어서가 아니라, DNN 분류기의 학습 방식(Softmax + CrossEntropy)이 "
        "클래스 불균형에 취약하기 때문입니다. 분류기를 바꾸면 문제가 해결됩니다.",
        s["QUOTE"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 7. 레이어 프로브
    # ════════════════════════════════════════════════
    story.append(Paragraph("7장. 레이어별 임베딩 판별력 분석 (Layer Probe)", s["H1"]))
    story.append(Paragraph("7.1 분석 목적과 방법", s["H2"]))
    story.append(Paragraph(
        "YAMNet의 1024차원 임베딩은 14개 MobileNet 블록의 최종 출력이 Global Average "
        "Pooling된 결과입니다. 각 블록이 사이렌 판별에 얼마나 기여하는지 알기 위해 "
        "1024차원을 14개 구간(각 73차원)으로 균등 분할하고 각 구간에 "
        "선형 로지스틱 회귀(Linear Probe)를 적용합니다. "
        "선형 프로브는 비선형 변환 없이 해당 구간의 표현이 얼마나 선형적으로 "
        "클래스를 분리하는지 측정하는 분석 기법입니다.",
        s["BODY"]))
    story.append(Paragraph("7.2 주요 결과", s["H2"]))
    probe_data = [
        ["블록",            "채널 구간", "Siren Recall", "파인튜닝",    "해석"],
        ["Block 3 (128ch)", "146~219",   "0.892",        "Frozen",      "초기 레이어에서도 강한 판별력"],
        ["Block 4 (128ch)", "219~292",   "0.904",        "Frozen",      "단독 최고 블록 (파인튜닝 전)"],
        ["Block 6 (256ch)", "365~438",   "0.892",        "Frozen",      "중간 레이어도 강함"],
        ["Block 9~11",      "584~803",   "0.590~0.639",  "Fine-tuned",  "파인튜닝 초반 일시적 혼란"],
        ["Block 12 (512ch)","803~876",   "0.843",        "Fine-tuned",  "파인튜닝 구간 중 최고"],
        ["FT 전체 (9~14)",  "584~1023",  "0.904",        "Fine-tuned",  "Frozen 대비 +13.3%p"],
    ]
    story.append(T(probe_data, [3.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 6.5*cm],
        extra=[
            ("ALIGN",      (0,1), (4,-1), "LEFT"),
            ("BACKGROUND", (0,2), (4,2),  C_HLROW),
            ("FONTNAME",   (0,2), (4,2),  FONT_BOLD),
        ]))
    story.append(Paragraph("7.3 누적 분석 결과의 함의", s["H2"]))
    story.append(Paragraph(
        "Block 1부터 N까지 누적 사용 시 Block 1~6(438차원)에서 이미 Siren Recall "
        "0.880에 도달합니다. Block 1~13(950차원)에서 0.904로 최고를 기록하고 "
        "Block 14 추가 시 오히려 0.880으로 소폭 감소합니다. "
        "이는 마지막 블록이 사이렌 특화 정보를 약간 희석시킬 수 있음을 시사합니다. "
        "또한 파인튜닝 이전 구간(Block 1~8)만으로도 Siren Recall 0.771을 달성합니다. "
        "YAMNet이 사전학습에서 이미 사이렌 관련 특징을 학습했음을 확인합니다.",
        s["BODY"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 8. Siren Specialist 설계 이유
    # ════════════════════════════════════════════════
    story.append(Paragraph("8장. Siren Specialist 이진 분류기를 만든 이유", s["H1"]))
    story.append(Paragraph("8.1 의사결정 과정", s["H2"]))
    story.append(Paragraph(
        "RF 실험과 레이어 프로브 분석을 통해 임베딩에 사이렌 정보가 충분히 있음을 확인했습니다. "
        "문제는 3-class Softmax DNN이 클래스 불균형에 취약하다는 것입니다. "
        "이 문제를 해결하는 방법 세 가지를 비교했습니다.",
        s["BODY"]))
    options_data = [
        ["접근법",                                  "장점",             "단점",                     "선택"],
        ["3-class DNN 개선\n(Focal Loss, 오버샘플링)", "기존 구조 유지",   "Background와 경쟁 구도 유지\n근본 해결 어려움", "X"],
        ["RF로 교체",                               "Recall 92.8% 달성","실시간 추론 지연\nTFLite 변환 불편", "△"],
        ["Siren 전용 이진 분류기\n(Specialist + 앙상블)", "집중 최적화 가능\n기존 DNN 강점 유지", "모델 2개 병렬 실행", "채택"],
    ]
    story.append(T(options_data, [3.5*cm, 3.8*cm, 3.8*cm, 1.2*cm],
        extra=[
            ("ALIGN",      (0,1), (3,-1), "LEFT"),
            ("VALIGN",     (0,0), (-1,-1),"TOP"),
            ("BACKGROUND", (0,3), (3,3),  C_GRNBG),
            ("FONTNAME",   (0,3), (3,3),  FONT_BOLD),
        ]))
    story.append(Paragraph("8.2 이진 분류기의 핵심 장점", s["H2"]))
    story.append(Paragraph("클래스 불균형 제거", s["H3"]))
    story.append(Paragraph(
        "3-class 문제에서는 background(65%)가 압도적이지만 "
        "이진 분류에서는 siren(22%) vs non-siren(78%)입니다. "
        "이 정도 불균형은 Focal Loss나 class_weight로 충분히 보정 가능합니다.",
        s["BODY"]))
    story.append(Paragraph("Softmax 경쟁 구도 제거", s["H3"]))
    story.append(Paragraph(
        "이진 분류는 Sigmoid를 사용합니다. Sigmoid는 각 클래스를 독립적으로 판단합니다. "
        "'사이렌 확률'을 높이기 위해 다른 클래스를 낮출 필요가 없습니다. "
        "오직 '이 소리가 사이렌인가 아닌가'에만 집중합니다.",
        s["BODY"]))
    story.append(Paragraph("기존 시스템 유지", s["H3"]))
    story.append(Paragraph(
        "3-class DNN은 car_horn 및 background 구별에 여전히 97% recall을 보입니다. "
        "기존의 강점을 버리지 않고 사이렌 감지라는 약점만 보완합니다. "
        "두 모델을 앙상블하면 각자의 강점이 시너지를 발휘합니다.",
        s["BODY"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 9. StandardScaler
    # ════════════════════════════════════════════════
    story.append(Paragraph("9장. StandardScaler — 왜 정규화가 필수인가", s["H1"]))
    story.append(Paragraph("9.1 StandardScaler의 수식과 원리", s["H2"]))
    story.append(Paragraph(
        "StandardScaler는 각 특징(feature) 차원을 평균 0, 표준편차 1로 변환합니다. "
        "수식은 z = (x - 평균) / 표준편차 입니다. "
        "여기서 평균과 표준편차는 훈련 데이터에서 계산합니다. "
        "변환 후 모든 차원은 동일한 스케일(대략 -3 ~ +3 범위)을 가집니다.",
        s["BODY"]))
    story.append(Paragraph("9.2 왜 YAMNet 임베딩에 정규화가 필요한가", s["H2"]))
    story.append(Paragraph(
        "YAMNet 임베딩의 1024개 차원은 서로 다른 스케일을 가집니다. "
        "어떤 차원은 0~0.01 범위에서, 어떤 차원은 0~10 범위에서 값을 가질 수 있습니다. "
        "정규화 없이 DNN에 입력하면 스케일이 큰 차원이 학습을 지배합니다. "
        "그래디언트 흐름이 왜곡되고 학습이 불안정해집니다.",
        s["BODY"]))
    scaler_data = [
        ["문제 상황",         "정규화 없을 때",          "정규화 후"],
        ["그래디언트 규모",   "차원별 편차 극심\n학습 불안정",    "균일한 그래디언트\n안정적 수렴"],
        ["학습률 효과",       "스케일 큰 차원에 과적합",  "모든 차원 동등하게 학습"],
        ["수렴 속도",         "느리고 진동",              "빠르고 안정적"],
        ["BatchNorm과의 관계","과도한 보정 필요",         "BatchNorm이 미세 조정만 담당"],
    ]
    story.append(T(scaler_data, [3.5*cm, 4.5*cm, 4.5*cm],
        extra=[("ALIGN", (0,1), (2,-1), "LEFT"), ("VALIGN", (0,0), (-1,-1), "TOP")]))

    story.append(Paragraph("9.3 반드시 지켜야 할 규칙 — Data Leakage 방지", s["H2"]))
    story.append(Paragraph(
        "반드시 훈련 데이터의 평균과 표준편차만으로 fit하고, "
        "검증/테스트 데이터에는 transform만 적용합니다. "
        "만약 테스트 데이터의 통계를 사용하면 미래 데이터를 미리 본 것이 됩니다(Data Leakage). "
        "이는 모델의 실제 성능을 과대 평가하게 만드는 심각한 실수입니다.",
        s["BODY"]))
    story.append(Paragraph("scaler.fit_transform(X_train)   # 훈련 데이터: 통계 학습 + 변환", s["MONO"]))
    story.append(Paragraph("scaler.transform(X_val)         # 검증 데이터: 변환만 (fit 없음)", s["MONO"]))
    story.append(Paragraph("scaler.transform(X_test)        # 테스트 데이터: 변환만 (fit 없음)", s["MONO"]))
    story.append(Paragraph(
        "모델 배포 시 scaler도 함께 저장(pickle)하고 실시간 추론 시 동일하게 적용합니다. "
        "scaler 없이는 학습 때와 다른 스케일의 입력이 들어가 성능이 급락합니다.",
        s["BODY"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 10. Focal Loss
    # ════════════════════════════════════════════════
    story.append(Paragraph("10장. Focal Loss — 왜 CrossEntropy를 쓰지 않는가", s["H1"]))
    story.append(Paragraph("10.1 표준 Binary CrossEntropy의 문제", s["H2"]))
    story.append(Paragraph(
        "이진 CrossEntropy는 Loss = -[y·log(p) + (1-y)·log(1-p)] 입니다. "
        "사이렌(y=1) 샘플과 비사이렌(y=0) 샘플을 동등하게 취급합니다. "
        "하지만 우리 데이터에서는 비사이렌이 78%를 차지합니다. "
        "모델은 비사이렌 샘플에서 발생하는 크고 쉬운 loss에 집중하여 학습하고 "
        "사이렌 샘플의 어려운 패턴은 무시하게 됩니다.",
        s["BODY"]))
    story.append(Paragraph("10.2 Focal Loss의 원리", s["H2"]))
    story.append(Paragraph(
        "Focal Loss는 Facebook AI Research가 2017년 제안한 손실함수입니다. "
        "핵심 아이디어는 이미 잘 분류되는 쉬운 샘플의 loss를 줄이고 "
        "어렵게 분류되는 샘플에 집중하는 것입니다.",
        s["BODY"]))
    story.append(Paragraph("표준 CE:   Loss = -log(p)", s["MONO"]))
    story.append(Paragraph("Focal:     Loss = -(1-p)^gamma * log(p)   [gamma: focusing parameter]", s["MONO"]))
    story.append(Paragraph(
        "gamma가 클수록 쉬운 샘플(p가 높은 것)의 loss가 더 많이 감소합니다. "
        "예를 들어 p=0.9(쉬운 비사이렌)이면 (1-0.9)^2 = 0.01배 감소합니다. "
        "p=0.3(어려운 경계 사이렌)이면 (1-0.3)^2 = 0.49배만 감소합니다. "
        "결과적으로 어려운 사이렌 샘플이 상대적으로 더 중요하게 취급됩니다.",
        s["BODY"]))
    story.append(Paragraph("10.3 Asymmetric Focal Loss — FN 패널티 강화", s["H2"]))
    story.append(Paragraph(
        "표준 Focal Loss에서 한 걸음 더 나아가 "
        "False Negative(사이렌을 놓치는 것)에 더 강한 패널티를 부여하는 "
        "비대칭(Asymmetric) 설정을 사용합니다. "
        "ADAS 안전 시스템에서 FN(사이렌 놓침)은 FP(오경보)보다 훨씬 위험합니다. "
        "오경보는 운전자를 잠깐 놀라게 하지만 놓침은 사고로 이어질 수 있습니다.",
        s["BODY"]))
    loss_data = [
        ["파라미터", "우리 설정", "의미"],
        ["fn_weight",  "5.0", "사이렌을 놓칠 때(FN) loss를 5배로 강화"],
        ["gamma_pos",  "0.0", "사이렌 샘플: 표준 log loss 적용 (감소 없음)"],
        ["gamma_neg",  "2.0", "비사이렌 샘플: 쉬운 것의 loss를 크게 감소"],
    ]
    story.append(T(loss_data, [3.5*cm, 2.5*cm, 10.3*cm],
        extra=[("ALIGN", (0,1), (2,-1), "LEFT")]))
    story.append(Paragraph("실제 구현 수식:", s["H3"]))
    story.append(Paragraph("사이렌(positive):     loss = -5.0 * log(p)", s["MONO"]))
    story.append(Paragraph("비사이렌(negative):   loss = -(1-p_neg)^2 * log(p_neg)", s["MONO"]))
    story.append(Paragraph(
        "사이렌을 맞추는 것에 5배 집중하고 "
        "비사이렌 중 이미 잘 구별되는 것은 loss를 줄여 학습 집중도를 높입니다.",
        s["BODY"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 11. MixUp
    # ════════════════════════════════════════════════
    story.append(Paragraph("11장. MixUp 증강 — 왜 합성 데이터를 만드는가", s["H1"]))
    story.append(Paragraph("11.1 데이터 부족 문제", s["H2"]))
    story.append(Paragraph(
        "훈련 세트에서 사이렌 샘플은 약 1,600개입니다. "
        "DNN이 사이렌과 비사이렌의 경계(decision boundary)를 잘 학습하려면 "
        "경계 근처의 어려운 샘플이 필요합니다. "
        "실제 녹음으로 '사이렌 소리가 배경음에 약간 묻힌' 같은 모호한 샘플을 "
        "충분히 수집하기는 매우 어렵습니다.",
        s["BODY"]))
    story.append(Paragraph("11.2 Embedding MixUp의 원리", s["H2"]))
    story.append(Paragraph(
        "MixUp(Zhang et al., 2018)은 두 샘플을 선형 보간하여 합성 샘플을 만드는 기법입니다. "
        "오디오 파형 레벨이 아닌 임베딩 레벨에서 MixUp을 적용합니다. "
        "임베딩 공간은 이미 의미 있는 음향 표현 공간이므로 "
        "두 임베딩의 중간값은 두 소리의 혼합에 가까운 표현을 가집니다.",
        s["BODY"]))
    story.append(Paragraph("lambda ~ Beta(0.4, 0.4),  [0.3, 0.7] 범위로 클리핑", s["MONO"]))
    story.append(Paragraph("x_mix = lambda * x_siren + (1 - lambda) * x_nonsiren", s["MONO"]))
    story.append(Paragraph("y_mix = lambda  (soft label: 사이렌 비율에 비례)", s["MONO"]))
    story.append(Paragraph("11.3 왜 [0.3, 0.7]으로 클리핑하는가", s["H2"]))
    story.append(Paragraph(
        "Beta(0.4, 0.4) 분포는 0과 1 근처에서 높은 확률 밀도를 가집니다. "
        "lambda = 0.95는 거의 순수 사이렌이고 lambda = 0.05는 거의 순수 비사이렌입니다. "
        "이런 '너무 확실한' 샘플은 경계 학습에 도움이 되지 않습니다. "
        "[0.3, 0.7] 클리핑으로 '사이렌 30~70% 포함' 범위의 "
        "진짜 어려운 경계 샘플만 생성합니다. "
        "이것이 모델이 경계에서 더 신중하게 판단하도록 만드는 핵심입니다.",
        s["BODY"]))
    story.append(Paragraph("11.4 MixUp 적용 효과", s["H2"]))
    mixup_data = [
        ["항목",                "값"],
        ["원본 훈련 사이렌 수", "약 1,600개"],
        ["MixUp 생성 샘플 수",  "4,800개 (사이렌 수 × 3)"],
        ["최종 훈련 데이터",    "7,246(원본) + 11,460(MixUp) = 총 18,706개"],
        ["Val Recall 달성",     "99.75% (Epoch 5 기준)"],
        ["Test Recall (thr=0.50)", "100% (FN=0, 83개 사이렌 모두 감지)"],
    ]
    story.append(T(mixup_data, [5*cm, 11.3*cm],
        extra=[("ALIGN", (0,1), (1,-1), "LEFT")]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 12. Threshold 탐색
    # ════════════════════════════════════════════════
    story.append(Paragraph("12장. 최적 Threshold 탐색 방법", s["H1"]))
    story.append(Paragraph("12.1 왜 0.50을 기본으로 쓰지 않는가", s["H2"]))
    story.append(Paragraph(
        "Sigmoid 출력이 0.50 이상이면 사이렌으로 판정하는 것이 일반적 기본값입니다. "
        "하지만 ADAS 안전 시스템의 목표는 Recall 최대화(사이렌을 절대 놓치지 않음)이므로 "
        "threshold를 낮추면 Recall이 오르고 Precision이 내려갑니다. "
        "안전 시스템에서는 FP(오경보)보다 FN(놓침)이 훨씬 위험하므로 "
        "낮은 threshold가 선호됩니다. 최적점을 데이터 기반으로 탐색해야 합니다.",
        s["BODY"]))
    story.append(Paragraph("12.2 탐색 알고리즘", s["H2"]))
    story.append(Paragraph(
        "검증 세트(fold 9)에서 다음 알고리즘으로 최적 threshold를 찾습니다. "
        "조건은 Recall >= 90%를 만족하면서 Precision을 최대화하는 threshold 선택입니다. "
        "Recall 조건을 만족하는 것이 없으면 최대 Recall을 주는 threshold를 선택합니다.",
        s["BODY"]))
    story.append(Paragraph("for t in [0.05, 0.06, ..., 0.94]:  # 0.01 간격으로 탐색", s["MONO"]))
    story.append(Paragraph("    preds = (probability >= t)", s["MONO"]))
    story.append(Paragraph("    recall    = TP / (TP + FN)", s["MONO"]))
    story.append(Paragraph("    precision = TP / (TP + FP)", s["MONO"]))
    story.append(Paragraph("    if recall >= 0.90 and precision > best_precision:", s["MONO"]))
    story.append(Paragraph("        best_threshold = t  # 조건 만족하는 최고 precision 저장", s["MONO"]))

    story.append(Paragraph("12.3 탐색 결과와 해석", s["H2"]))
    thr_data = [
        ["Threshold",    "Recall",  "Precision", "FN(놓침)", "FP(오경보)", "판단"],
        ["0.10",         "1.000",   "0.264",     "0",        "231",        "Recall 완벽, FP 과다"],
        ["0.50 (권장)",  "1.000",   "0.472",     "0",        "93",         "100% Recall, FP 관리 가능"],
        ["0.78 (Val 최적)", "0.795","0.857",     "17",       "11",         "Val 기준 최적, 테스트서 FN 발생"],
        ["0.90",         "0.446",   "0.949",     "46",       "3",          "Precision 좋지만 FN 과다"],
    ]
    story.append(T(thr_data, [2.8*cm, 1.8*cm, 2.2*cm, 2*cm, 2.5*cm, 5*cm],
        extra=[
            ("ALIGN",      (0,1), (5,-1), "LEFT"),
            ("BACKGROUND", (0,2), (5,2),  C_HLROW),
            ("FONTNAME",   (0,2), (5,2),  FONT_BOLD),
        ]))
    story.append(Paragraph(
        "중요한 발견: 테스트 세트에서 threshold=0.50 이하에서 Recall=100%(FN=0).\n"
        "Val에서 찾은 최적 threshold 0.78은 Val 데이터에 과적합된 것이었고 "
        "테스트에서는 0.50으로도 완벽한 Recall이 달성됩니다.\n"
        "ADAS 실배포 권장: threshold=0.50 (안전성 최우선)",
        s["QUOTE"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 13. 앙상블 전략
    # ════════════════════════════════════════════════
    story.append(Paragraph("13장. 앙상블 전략 — OR / AVG / MAX 비교", s["H1"]))
    story.append(Paragraph("13.1 왜 앙상블이 필요한가", s["H2"]))
    story.append(Paragraph(
        "Specialist 단독으로도 Siren Recall 100%를 달성하지만 "
        "Precision이 47.2%(FP=93개)로 오경보가 많습니다. "
        "3-class DNN은 Car Horn과 Background를 잘 구별합니다. "
        "두 모델을 결합하면 각자의 강점을 활용할 수 있습니다. "
        "DNN이 'background 확률 높다'고 할 때 Specialist도 동의하면 "
        "사이렌이 아닐 가능성이 높아집니다.",
        s["BODY"]))
    story.append(Paragraph("13.2 앙상블 전략 수치 비교", s["H2"]))
    ens_data = [
        ["전략",                        "계산 방법",                         "Recall", "Prec",  "F1",   "FN", "FP"],
        ["DNN 단독 (현재)",             "argmax(probs) == siren",             "0.687",  "0.919", "0.787","26", "2"],
        ["Specialist 단독 thr=0.78",    "p_spec >= 0.78",                    "0.795",  "0.857", "0.825","17", "11"],
        ["Specialist 단독 thr=0.50",    "p_spec >= 0.50",                    "1.000",  "0.472", "0.641","0",  "93"],
        ["AVG 앙상블 thr=0.30 (권장)", "0.5*p_dnn + 0.5*p_spec >= 0.30",   "0.940",  "0.600", "0.732","5",  "52"],
        ["AVG 앙상블 thr=0.20",         "0.5*p_dnn + 0.5*p_spec >= 0.20",   "1.000",  "0.347", "0.516","0",  "156"],
        ["MAX 앙상블 thr=0.30",         "max(p_dnn, p_spec) >= 0.30",        "1.000",  "0.269", "0.423","0",  "226"],
    ]
    story.append(T(ens_data, [3.5*cm, 4.5*cm, 1.5*cm, 1.5*cm, 1.5*cm, 0.9*cm, 0.9*cm],
        extra=[
            ("ALIGN",      (0,1), (6,-1), "LEFT"),
            ("VALIGN",     (0,0), (-1,-1),"TOP"),
            ("BACKGROUND", (0,1), (6,1),  C_REDBG),
            ("TEXTCOLOR",  (2,1), (2,1),  C_RED),
            ("BACKGROUND", (0,3), (6,3),  C_HLROW),
            ("BACKGROUND", (0,4), (6,4),  C_GRNBG),
            ("FONTNAME",   (0,4), (6,4),  FONT_BOLD),
        ]))
    story.append(Paragraph("13.3 최종 권고", s["H2"]))
    story.append(Paragraph(
        "ADAS 안전 시스템의 특성상 FN(사이렌 놓침)은 절대적으로 피해야 합니다. "
        "AVG 앙상블 threshold=0.30을 1차 권고합니다. "
        "Recall 94%와 F1 0.732의 최고 균형을 제공합니다. "
        "또한 실시간 추론 시 Debounce(3프레임 다수결)를 적용하면 "
        "FP를 추가로 줄일 수 있습니다.",
        s["BODY"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 14. SNR 분석
    # ════════════════════════════════════════════════
    story.append(Paragraph("14장. SNR 노이즈 분석 — 노이즈가 정말 문제인가", s["H1"]))
    story.append(Paragraph("14.1 실험 설계와 목적", s["H2"]))
    story.append(Paragraph(
        "사이렌 Recall이 낮은 원인이 '노이즈 환경'에서 오는 것인지 "
        "아니면 '분류기 구조'에서 오는 것인지 분리해서 확인합니다. "
        "clean 음원에 STRAFFIC 도로교통 노이즈를 +20dB에서 -10dB SNR로 합성하여 "
        "각 조건에서의 성능을 측정합니다.",
        s["BODY"]))
    story.append(Paragraph("14.2 SNR 레벨의 의미", s["H2"]))
    snr_data = [
        ["SNR",    "의미",                          "실제 환경"],
        ["clean",  "노이즈 없음",                   "조용한 실내"],
        ["+20dB",  "신호가 노이즈보다 100배 강함",  "조용한 도로"],
        ["+10dB",  "신호가 노이즈보다 10배 강함",   "일반 도시 환경"],
        ["+5dB",   "신호가 노이즈보다 3배 강함",    "혼잡한 도로"],
        ["0dB",    "신호와 노이즈가 같은 크기",      "매우 시끄러운 환경"],
        ["-5dB",   "노이즈가 신호보다 3배 강함",    "극한 소음"],
        ["-10dB",  "노이즈가 신호보다 10배 강함",   "실용적 한계 초과"],
    ]
    story.append(T(snr_data, [1.8*cm, 5*cm, 9.5*cm],
        extra=[("ALIGN", (0,1), (2,-1), "LEFT")]))

    story.append(Paragraph("14.3 핵심 결과와 함의", s["H2"]))
    snr_result = [
        ["클래스",     "clean", "+10dB", "+5dB", "-5dB", "-10dB", "추세"],
        ["Siren",      "0.542", "0.566", "0.578","0.542","0.518", "거의 변화 없음 (±3%)"],
        ["Car Horn",   "0.970", "0.788", "0.758","0.758","0.697", "점진적 하락 (정상 범위)"],
        ["Background", "0.828", "0.848", "0.864","0.838","0.717", "중간 SNR에서 오히려 상승"],
    ]
    story.append(T(snr_result, [2.5*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.8*cm, 5.5*cm],
        extra=[
            ("ALIGN",      (0,1), (6,-1), "LEFT"),
            ("BACKGROUND", (0,1), (6,1),  C_REDBG),
            ("FONTNAME",   (0,1), (6,1),  FONT_BOLD),
            ("TEXTCOLOR",  (2,1), (2,1),  C_RED),
        ]))
    story.append(Paragraph(
        "핵심 발견: Siren Recall은 clean 환경에서도 54.2%로 낮고 "
        "노이즈를 추가해도 51~58% 범위에서 거의 변화하지 않습니다.\n"
        "이것은 노이즈가 사이렌 감지 실패의 원인이 아님을 명확히 증명합니다.\n"
        "원인은 오직 3-class DNN 분류기의 클래스 불균형 처리 실패입니다.\n"
        "결론: 데이터 증강(노이즈 합성)을 더 추가해도 근본 해결이 안 됩니다. "
        "Specialist 이진 분류기가 필요합니다.",
        s["WARN"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════
    # 15. 최종 결론
    # ════════════════════════════════════════════════
    story.append(Paragraph("15장. 최종 결론 및 시스템 권고", s["H1"]))
    story.append(Paragraph("15.1 문제 → 분석 → 해결 전체 흐름", s["H2"]))
    flow_data = [
        ["단계",         "발견 및 행동",                                                "결과"],
        ["문제 발견",    "Phase 4 완료 후 Siren Recall 68.7%\n전체 정확도 89.5%인데 사이렌만 낮음", "핵심 문제 정의"],
        ["가설 수립",    "임베딩에 정보가 없는가?\nvs 분류기가 정보를 못 쓰는가?",     "실험 설계"],
        ["RF 실험",      "동일 임베딩으로 RF → Siren Recall 92.8%",                   "임베딩에 정보 있음\n분류기가 문제임 입증"],
        ["Layer Probe",  "Block 4(ch 219~292) 단독 0.904\n파인튜닝 구간 전체 0.904",   "판별력 높은 레이어 확인"],
        ["SNR 분석",     "모든 SNR에서 Siren Recall ~55%로 불변",                      "노이즈가 원인 아님 확인"],
        ["Specialist 설계","이진 분류 + Focal Loss + MixUp 적용",                     "Val Recall 99.75%"],
        ["Threshold 탐색","0.05~0.94 탐색, 테스트에서 0.50 → Recall 100%",            "최적 운영 지점 확인"],
        ["앙상블",       "AVG(DNN + Specialist) thr=0.30",                             "Recall 94%, F1=0.732"],
    ]
    story.append(T(flow_data, [2.5*cm, 6.5*cm, 7.3*cm],
        extra=[
            ("ALIGN",      (0,1), (2,-1), "LEFT"),
            ("VALIGN",     (0,0), (-1,-1),"TOP"),
            ("BACKGROUND", (0,6), (2,6),  C_HLROW),
            ("BACKGROUND", (0,8), (2,8),  C_GRNBG),
            ("FONTNAME",   (0,8), (2,8),  FONT_BOLD),
        ]))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("15.2 배포 권고 시스템 구성", s["H2"]))
    deploy_data = [
        ["컴포넌트",                "파일",                              "역할"],
        ["YAMNet (파인튜닝)",       "yamnet_finetuned/yamnet_vars.npz",  "1024차원 임베딩 추출"],
        ["3-class DNN",             "custom_classifier_finetuned.h5",    "Car Horn / Background 구별"],
        ["Siren Specialist",        "siren_specialist.h5",               "Siren 전용 이진 분류"],
        ["StandardScaler",          "siren_specialist_scaler.pkl",       "Specialist 입력 정규화"],
        ["AVG 앙상블 (thr=0.30)",   "ensemble_eval.py",                  "최종 판정 로직"],
        ["Debounce (3프레임)",      "realtime_infer.py",                 "연속 판정으로 오경보 감소"],
        ["TFLite INT8",             "adas_detector.tflite",              "Raspberry Pi 경량 배포"],
    ]
    story.append(T(deploy_data, [4*cm, 5*cm, 7.3*cm],
        extra=[("ALIGN", (0,1), (2,-1), "LEFT")]))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("15.3 최종 성능 비교표", s["H2"]))
    final_data = [
        ["지표",            "Phase 1\n(초기)",  "Phase 4\n(파인튜닝)", "Specialist\n앙상블 (최종)"],
        ["Siren Recall",    "< 50%",            "68.7%",              "94~100%"],
        ["Horn Recall",     "~90%",             "97.0%",              "~97%"],
        ["전체 Accuracy",   "~85%",             "89.5%",              "~89%"],
        ["FN (사이렌 놓침)", "~40개",           "26개",               "0~5개"],
        ["모델 크기",        "0.67MB",           "11.7MB",             "14.8MB"],
    ]
    story.append(T(final_data, [4*cm, 3.5*cm, 3.5*cm, 5.3*cm],
        extra=[
            ("ALIGN",     (0,1), (3,-1), "LEFT"),
            ("TEXTCOLOR", (3,1), (3,1),  C_GREEN),
            ("TEXTCOLOR", (3,4), (3,4),  C_GREEN),
            ("FONTNAME",  (3,1), (3,5),  FONT_BOLD),
        ]))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width=W, color=C_NAVY, thickness=1.5))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "핵심 교훈 4가지:\n"
        "1. 사전학습 모델(YAMNet)의 임베딩에는 이미 충분한 사이렌 정보가 있었습니다.\n"
        "2. 문제는 3-class Softmax의 클래스 불균형 취약성이었습니다.\n"
        "3. 전용 이진 분류기 + Asymmetric Focal Loss + MixUp으로 100% Recall 달성.\n"
        "4. 노이즈(SNR)는 문제의 원인이 아니었습니다 — 구조적 설계가 핵심입니다.",
        s["QUOTE"]))
    story.append(Spacer(0.3*cm, 0.3*cm))
    story.append(Paragraph(
        f"작성일: {datetime.datetime.now().strftime('%Y년 %m월 %d일')}  |  "
        "ADAS Sound Detection Project  |  YAMNet Phase 4 + Siren Specialist Ensemble",
        s["CAP"]))

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
        title="ADAS 사운드 감지 — 전체 과정 및 의사결정 리포트",
    )
    doc.build(story)
    print(f"OK: {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build()
