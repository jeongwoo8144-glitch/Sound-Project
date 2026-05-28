"""
car_horn 오디오 증강 기법 설명 PDF 생성 스크립트
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

import os

# ── 폰트 등록 ──────────────────────────────────────────────────
pdfmetrics.registerFont(TTFont("MG",  "/mnt/c/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("MGB", "/mnt/c/Windows/Fonts/malgunbd.ttf"))

# ── 색상 팔레트 ────────────────────────────────────────────────
C_NAVY   = colors.HexColor("#1F4E79")
C_BLUE   = colors.HexColor("#2E75B6")
C_LBLUE  = colors.HexColor("#D6E4F0")
C_LLBLUE = colors.HexColor("#EBF5FB")
C_GREEN  = colors.HexColor("#1E8449")
C_LGREEN = colors.HexColor("#D5F5E3")
C_ORANGE = colors.HexColor("#D35400")
C_LORANG = colors.HexColor("#FDEBD0")
C_RED    = colors.HexColor("#922B21")
C_LRED   = colors.HexColor("#FADBD8")
C_GRAY   = colors.HexColor("#5D6D7E")
C_LGRAY  = colors.HexColor("#F2F3F4")
C_PURPLE = colors.HexColor("#6C3483")
C_LPURP  = colors.HexColor("#E8DAEF")
C_TEAL   = colors.HexColor("#117A65")
C_LTEAL  = colors.HexColor("#D1F2EB")
C_DARK   = colors.HexColor("#1C2833")
C_CODE_BG = colors.HexColor("#F8F9FA")
C_CODE_BORDER = colors.HexColor("#DEE2E6")

# ── 스타일 ─────────────────────────────────────────────────────
def make_styles():
    s = {}
    s["title"] = ParagraphStyle("title", fontName="MGB", fontSize=22,
        textColor=C_NAVY, spaceAfter=4, spaceBefore=0, alignment=TA_CENTER)
    s["subtitle"] = ParagraphStyle("subtitle", fontName="MG", fontSize=12,
        textColor=C_BLUE, spaceAfter=16, alignment=TA_CENTER)
    s["h1"] = ParagraphStyle("h1", fontName="MGB", fontSize=15,
        textColor=C_NAVY, spaceBefore=18, spaceAfter=6,
        borderPad=0, leading=20)
    s["h2"] = ParagraphStyle("h2", fontName="MGB", fontSize=12,
        textColor=C_BLUE, spaceBefore=12, spaceAfter=4, leading=16)
    s["h3"] = ParagraphStyle("h3", fontName="MGB", fontSize=10.5,
        textColor=C_GRAY, spaceBefore=8, spaceAfter=3, leading=14)
    s["body"] = ParagraphStyle("body", fontName="MG", fontSize=10,
        textColor=C_DARK, leading=16, spaceAfter=4, alignment=TA_JUSTIFY)
    s["body_left"] = ParagraphStyle("body_left", fontName="MG", fontSize=10,
        textColor=C_DARK, leading=16, spaceAfter=4, alignment=TA_LEFT)
    s["bullet"] = ParagraphStyle("bullet", fontName="MG", fontSize=10,
        textColor=C_DARK, leading=16, spaceAfter=3,
        leftIndent=12, firstLineIndent=-12)
    s["code"] = ParagraphStyle("code", fontName="MG", fontSize=8.5,
        textColor=C_DARK, leading=13, spaceAfter=0,
        leftIndent=0, backColor=C_CODE_BG)
    s["caption"] = ParagraphStyle("caption", fontName="MG", fontSize=8.5,
        textColor=C_GRAY, spaceAfter=6, alignment=TA_CENTER)
    s["tag"] = ParagraphStyle("tag", fontName="MGB", fontSize=9,
        textColor=colors.white, spaceAfter=0, alignment=TA_CENTER)
    s["small"] = ParagraphStyle("small", fontName="MG", fontSize=8.5,
        textColor=C_GRAY, leading=12, spaceAfter=2)
    s["note"] = ParagraphStyle("note", fontName="MG", fontSize=9,
        textColor=C_GRAY, leading=13, spaceAfter=4,
        leftIndent=8, borderPad=0)
    return s

ST = make_styles()

W, H = A4
MARGIN = 20 * mm
CONTENT_W = W - 2 * MARGIN

# ── 헬퍼 ───────────────────────────────────────────────────────
def hr(color=C_BLUE, thickness=0.8):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceAfter=6, spaceBefore=2)

def sp(h=4):
    return Spacer(1, h * mm)

def code_block(raw, lang_label="Python"):
    lines = raw.strip("\n").split("\n")
    parts = []
    for line in lines:
        stripped = line.lstrip(" ")
        n_sp = len(line) - len(stripped)
        indent = "&#160;" * n_sp
        escaped = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append((indent + escaped) if escaped else "&#160;")
    inner = "<br/>".join(parts)
    label_row = [[Paragraph(lang_label, ST["tag"])]]
    label_tbl = Table(label_row, colWidths=[CONTENT_W])
    label_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_BLUE),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    code_para = Paragraph(inner, ST["code"])
    code_row = [[code_para]]
    code_tbl = Table(code_row, colWidths=[CONTENT_W])
    code_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CODE_BG),
        ("BOX", (0,0), (-1,-1), 0.8, C_CODE_BORDER),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
    ]))
    return [label_tbl, code_tbl, sp(2)]

def section_header(num, title, color=C_NAVY):
    data = [[Paragraph(f"{num}. {title}", ParagraphStyle(
        "sh", fontName="MGB", fontSize=13, textColor=colors.white,
        leading=17, alignment=TA_LEFT
    ))]]
    tbl = Table(data, colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), color),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("ROUNDEDCORNERS", [3,3,3,3]),
    ]))
    return [sp(2), tbl, sp(3)]

def info_box(text, bg=C_LLBLUE, border=C_BLUE):
    data = [[Paragraph(text, ST["body_left"])]]
    tbl = Table(data, colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("BOX", (0,0), (-1,-1), 1.2, border),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
    ]))
    return [tbl, sp(2)]

def aug_card(tag_text, tag_color, title, desc_rows, reason_rows, col_w=None):
    if col_w is None:
        col_w = [18*mm, CONTENT_W - 18*mm]
    tag_cell = Paragraph(tag_text, ParagraphStyle(
        "tc", fontName="MGB", fontSize=9, textColor=colors.white,
        alignment=TA_CENTER, leading=12
    ))
    title_cell = Paragraph(title, ParagraphStyle(
        "tc2", fontName="MGB", fontSize=11, textColor=C_DARK, leading=14
    ))
    hdr_data = [[tag_cell, title_cell]]
    hdr = Table(hdr_data, colWidths=col_w)
    hdr.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), tag_color),
        ("BACKGROUND", (1,0), (1,0), C_LGRAY),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (0,0), 4),
        ("LEFTPADDING", (1,0), (1,0), 10),
        ("BOX", (0,0), (-1,-1), 0.5, C_GRAY),
    ]))

    desc_cells = [Paragraph(d, ST["body"]) for d in desc_rows]
    reason_label = Paragraph("<b>사용 이유</b>", ParagraphStyle(
        "rl", fontName="MGB", fontSize=9.5, textColor=tag_color, leading=13
    ))
    reason_cells = [Paragraph(r, ST["body"]) for r in reason_rows]

    body_items = desc_cells + [sp(1), reason_label] + reason_cells
    body_data = [[item] for item in body_items]
    body_tbl = Table([[item] for item in [Spacer(1,0)]], colWidths=[CONTENT_W])

    # Flatten into single cell
    from reportlab.platypus import ListFlowable
    inner = []
    for d in desc_cells:
        inner.append(d)
    inner.append(sp(1))
    inner.append(reason_label)
    for r in reason_cells:
        inner.append(r)

    from reportlab.platypus import FrameBreak
    body_para_combined = desc_cells + [sp(1), reason_label] + reason_cells

    body_row_data = [[item] for item in body_para_combined]
    # Use a simple table with all items stacked
    body_tbl2 = Table([[p] for p in body_para_combined], colWidths=[CONTENT_W])
    body_tbl2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.white),
        ("TOPPADDING", (0,0), (-1,-1), 1),
        ("BOTTOMPADDING", (0,0), (-1,-1), 1),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("BOX", (0,0), (-1,-1), 0.5, C_GRAY),
        ("LINEABOVE", (0,0), (-1,0), 0, colors.white),
    ]))

    return [hdr, body_tbl2, sp(3)]


# ── 페이지 번호 콜백 ───────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("MG", 8)
    canvas.setFillColor(C_GRAY)
    canvas.drawRightString(W - MARGIN, 12*mm, f"{doc.page}")
    canvas.drawString(MARGIN, 12*mm, "ADAS Sound Detector — car_horn 데이터 증강 기법 보고서")
    canvas.setStrokeColor(C_LBLUE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 14*mm, W - MARGIN, 14*mm)
    canvas.restoreState()


# ── 본문 구성 ──────────────────────────────────────────────────
def build_story():
    story = []

    # ── 표지 ─────────────────────────────────────────
    story.append(sp(20))
    story.append(Paragraph("ADAS Sound Detector", ST["subtitle"]))
    story.append(Paragraph("car_horn 데이터 증강 기법 보고서", ST["title"]))
    story.append(sp(3))
    story.append(hr(C_BLUE, 1.5))
    story.append(sp(2))
    story.append(Paragraph(
        "클래스 불균형 해소를 위한 오디오 증강 기법 설명 및 구현 코드",
        ST["subtitle"]))
    story.append(sp(8))

    # 요약 박스
    summary_data = [
        [Paragraph("<b>증강 배경</b>", ParagraphStyle("k", fontName="MGB", fontSize=10, textColor=C_NAVY, leading=14)),
         Paragraph("UrbanSound8K 원본 데이터에서 siren 929개 vs car_horn 429개 (2.17배 불균형)", ST["body_left"])],
        [Paragraph("<b>증강 목표</b>", ParagraphStyle("k", fontName="MGB", fontSize=10, textColor=C_NAVY, leading=14)),
         Paragraph("fold별 car_horn 수를 siren과 동일하게 맞춤 → 527개 신규 파일 생성", ST["body_left"])],
        [Paragraph("<b>적용 범위</b>", ParagraphStyle("k", fontName="MGB", fontSize=10, textColor=C_NAVY, leading=14)),
         Paragraph("clean 조건 증강 후 SNR 4개 조건(+10, +5, 0, -5 dB)에도 동일 적용 → 총 2,635개 파일", ST["body_left"])],
        [Paragraph("<b>기법 수</b>", ParagraphStyle("k", fontName="MGB", fontSize=10, textColor=C_NAVY, leading=14)),
         Paragraph("12가지 변환 (Time Stretch 4종, Pitch Shift 4종, Gain 2종, Combo 2종)", ST["body_left"])],
    ]
    summary_tbl = Table(summary_data, colWidths=[32*mm, CONTENT_W - 32*mm])
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), C_LBLUE),
        ("BACKGROUND", (1,0), (1,-1), colors.white),
        ("BOX", (0,0), (-1,-1), 1.2, C_BLUE),
        ("INNERGRID", (0,0), (-1,-1), 0.4, C_LBLUE),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(summary_tbl)
    story.append(PageBreak())

    # ── 1. 증강이 필요한 이유 ─────────────────────────
    story += section_header("1", "증강이 필요한 이유 — 클래스 불균형 문제")

    story.append(Paragraph(
        "딥러닝 분류 모델은 훈련 데이터의 클래스 분포가 불균일할 때 다수 클래스에 편향되는 경향이 있습니다. "
        "본 프로젝트에서 사용한 UrbanSound8K 데이터셋은 원본 수집 단계에서 클래스별 샘플 수가 균등하지 않습니다.",
        ST["body"]))

    imbalance_data = [
        [Paragraph("클래스", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("clean 샘플 수", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("전체(×5 SNR)", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("비율", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER))],
        [Paragraph("siren", ST["body_left"]),      Paragraph("929",  ST["body_left"]), Paragraph("4,645", ST["body_left"]), Paragraph("기준 (1.00×)", ST["body_left"])],
        [Paragraph("car_horn", ST["body_left"]),   Paragraph("429",  ST["body_left"]), Paragraph("2,145", ST["body_left"]),
         Paragraph("<font color='#922B21'><b>0.46× (2.17배 부족)</b></font>", ST["body_left"])],
        [Paragraph("background", ST["body_left"]), Paragraph("2,000",ST["body_left"]), Paragraph("2,000", ST["body_left"]), Paragraph("clean 전용", ST["body_left"])],
    ]
    imb_tbl = Table(imbalance_data, colWidths=[38*mm, 38*mm, 38*mm, CONTENT_W - 114*mm])
    imb_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_NAVY),
        ("BACKGROUND", (0,2), (-1,2), C_LRED),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_LRED, C_LGRAY]),
        ("BOX", (0,0), (-1,-1), 1, C_BLUE),
        ("INNERGRID", (0,0), (-1,-1), 0.4, C_LBLUE),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
    ]))
    story.append(imb_tbl)
    story.append(sp(2))

    story.append(Paragraph(
        "car_horn은 자동차 경적 소리로, 도로 환경에서 중요한 경보 신호입니다. "
        "샘플 수 부족은 모델이 경적을 제대로 인식하지 못하는 원인이 되므로, "
        "기존 car_horn 샘플에 다양한 변환을 적용하여 샘플 수를 siren과 동일하게 맞추었습니다.",
        ST["body"]))

    story += info_box(
        "<b>fold5 예외 처리:</b> fold5는 원본 데이터에서 car_horn(98개)이 siren(71개)보다 많아 "
        "증강 없이 그대로 유지하였습니다. fold1~4, fold6~10의 9개 fold에서 총 527개를 증강하였습니다.",
        C_LGREEN, C_GREEN
    )

    # ── 2. 증강 기법 개요 ─────────────────────────────
    story += section_header("2", "증강 기법 개요")
    story.append(PageBreak())

    story.append(Paragraph(
        "총 12가지 변환을 4개 카테고리로 분류합니다. 각 fold에서 필요한 샘플 수만큼 "
        "원본 파일을 순환하면서 12개 변환을 순서대로 적용합니다.",
        ST["body"]))
    story.append(sp(2))

    overview_data = [
        [Paragraph("카테고리", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("변환 태그", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("변환 파라미터", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("핵심 효과", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER))],
        [Paragraph("Time Stretch", ST["body_left"]), Paragraph("ts085, ts090\nts110, ts115", ST["body_left"]),
         Paragraph("rate = 0.85, 0.90\n1.10, 1.15", ST["body_left"]),
         Paragraph("재생 속도 변화, 음높이 유지", ST["body_left"])],
        [Paragraph("Pitch Shift", ST["body_left"]), Paragraph("ps_m3, ps_m2\nps_p2, ps_p3", ST["body_left"]),
         Paragraph("n_steps = -3, -2\n+2, +3", ST["body_left"]),
         Paragraph("음높이 변화, 속도 유지", ST["body_left"])],
        [Paragraph("Gain", ST["body_left"]), Paragraph("gain_m4\ngain_p4", ST["body_left"]),
         Paragraph("-4 dB\n+4 dB", ST["body_left"]),
         Paragraph("볼륨 증감, 파형 형태 유지", ST["body_left"])],
        [Paragraph("Combo", ST["body_left"]), Paragraph("combo1\ncombo2", ST["body_left"]),
         Paragraph("ts0.90 + ps+2\nts1.10 + ps-2", ST["body_left"]),
         Paragraph("복합 변환으로 다양성 극대화", ST["body_left"])],
    ]
    ov_col = [30*mm, 32*mm, 38*mm, CONTENT_W - 100*mm]
    ov_tbl = Table(overview_data, colWidths=ov_col)
    ov_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_NAVY),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_LLBLUE, colors.white, C_LGREEN, C_LORANG]),
        ("BOX", (0,0), (-1,-1), 1, C_BLUE),
        ("INNERGRID", (0,0), (-1,-1), 0.4, C_LBLUE),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(ov_tbl)
    story.append(sp(3))

    # ── 3. 기법별 상세 설명 ───────────────────────────
    story += section_header("3", "기법별 상세 설명")

    # 3.1 Time Stretch
    story.append(Paragraph("3.1 Time Stretch (시간 신축)", ST["h2"]))
    story.append(hr(C_BLUE, 0.4))

    story.append(Paragraph(
        "Time Stretch는 오디오 신호의 <b>재생 속도를 변경하되 음높이(pitch)는 유지</b>하는 변환입니다. "
        "librosa의 <b>phase vocoder</b> 알고리즘을 사용하여 STFT(Short-Time Fourier Transform) "
        "영역에서 위상을 조정하는 방식으로 구현됩니다.",
        ST["body"]))
    story.append(sp(1))

    ts_data = [
        [Paragraph("태그", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("rate 값", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("효과", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("실제 적용 결과 (4초 파일 기준)", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER))],
        [Paragraph("ts085", ST["small"]), Paragraph("0.85×", ST["small"]),
         Paragraph("느리게 (15% 감속)", ST["small"]), Paragraph("4.71초 → 4초로 잘림", ST["small"])],
        [Paragraph("ts090", ST["small"]), Paragraph("0.90×", ST["small"]),
         Paragraph("느리게 (10% 감속)", ST["small"]), Paragraph("4.44초 → 4초로 잘림", ST["small"])],
        [Paragraph("ts110", ST["small"]), Paragraph("1.10×", ST["small"]),
         Paragraph("빠르게 (10% 가속)", ST["small"]), Paragraph("3.64초 → 0.36초 zero padding", ST["small"])],
        [Paragraph("ts115", ST["small"]), Paragraph("1.15×", ST["small"]),
         Paragraph("빠르게 (15% 가속)", ST["small"]), Paragraph("3.48초 → 0.52초 zero padding", ST["small"])],
    ]
    ts_col = [18*mm, 18*mm, 38*mm, CONTENT_W - 74*mm]
    ts_tbl = Table(ts_data, colWidths=ts_col)
    ts_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_BLUE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_LLBLUE, colors.white, C_LLBLUE, colors.white]),
        ("BOX", (0,0), (-1,-1), 0.8, C_BLUE),
        ("INNERGRID", (0,0), (-1,-1), 0.3, C_LBLUE),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("ALIGN", (0,0), (1,-1), "CENTER"),
    ]))
    story.append(ts_tbl)
    story.append(sp(2))

    story.append(Paragraph("<b>사용 이유:</b>", ST["h3"]))
    ts_reasons = [
        "• <b>현실 다양성 반영:</b> 실제 도로에서 경적 소리는 차량 속도, 마이크 거리, "
        "녹음 환경에 따라 재생 시간이 달라집니다. 느린/빠른 버전을 생성함으로써 다양한 녹음 조건을 시뮬레이션합니다.",
        "• <b>음높이 보존:</b> 경적 소리의 핵심 식별 특징은 주파수(음높이)입니다. "
        "Time Stretch는 속도만 바꾸고 주파수를 유지하므로 클래스 레이블 유효성이 보장됩니다.",
        "• <b>±15% 범위 선택:</b> 너무 극단적인 변환(±30% 이상)은 소리가 부자연스러워져 "
        "오히려 학습에 해가 됩니다. ±10~15% 범위는 자연스러운 변동 범위 내에 있습니다.",
    ]
    for r in ts_reasons:
        story.append(Paragraph(r, ST["bullet"]))
    story.append(sp(2))

    # 3.2 Pitch Shift
    story.append(Paragraph("3.2 Pitch Shift (음높이 변조)", ST["h2"]))
    story.append(hr(C_BLUE, 0.4))

    story.append(Paragraph(
        "Pitch Shift는 오디오의 <b>음높이(주파수)를 변경하되 재생 시간은 유지</b>하는 변환입니다. "
        "내부적으로 Time Stretch 후 리샘플링하는 방식으로 구현되며, "
        "반음(semitone) 단위로 지정합니다. 1 semitone = 약 6% 주파수 변화입니다.",
        ST["body"]))
    story.append(sp(1))

    ps_data = [
        [Paragraph("태그", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("n_steps", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("주파수 변화율", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("효과", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER))],
        [Paragraph("ps_m3", ST["small"]), Paragraph("-3", ST["small"]),
         Paragraph("×0.841 (-15.9%)", ST["small"]), Paragraph("낮은 음의 경적 (대형 트럭 등)", ST["small"])],
        [Paragraph("ps_m2", ST["small"]), Paragraph("-2", ST["small"]),
         Paragraph("×0.891 (-10.9%)", ST["small"]), Paragraph("약간 낮은 음의 경적", ST["small"])],
        [Paragraph("ps_p2", ST["small"]), Paragraph("+2", ST["small"]),
         Paragraph("×1.122 (+12.2%)", ST["small"]), Paragraph("약간 높은 음의 경적", ST["small"])],
        [Paragraph("ps_p3", ST["small"]), Paragraph("+3", ST["small"]),
         Paragraph("×1.189 (+18.9%)", ST["small"]), Paragraph("높은 음의 경적 (소형차 등)", ST["small"])],
    ]
    ps_col = [18*mm, 18*mm, 38*mm, CONTENT_W - 74*mm]
    ps_tbl = Table(ps_data, colWidths=ps_col)
    ps_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_GREEN),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_LGREEN, colors.white, C_LGREEN, colors.white]),
        ("BOX", (0,0), (-1,-1), 0.8, C_GREEN),
        ("INNERGRID", (0,0), (-1,-1), 0.3, C_LGREEN),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("ALIGN", (0,0), (1,-1), "CENTER"),
    ]))
    story.append(ps_tbl)
    story.append(sp(2))

    story.append(Paragraph("<b>사용 이유:</b>", ST["h3"]))
    ps_reasons = [
        "• <b>차종 다양성 반영:</b> 자동차 경적의 음높이는 차종마다 다릅니다. "
        "대형 트럭과 소형 승용차는 서로 다른 주파수 대역의 경적을 사용합니다. "
        "Pitch Shift로 다양한 차종의 경적을 시뮬레이션할 수 있습니다.",
        "• <b>마이크 특성 보상:</b> 스마트폰 마이크는 모델별로 주파수 응답 특성이 다릅니다. "
        "음높이 변조는 다양한 마이크 환경에서의 녹음을 간접적으로 재현합니다.",
        "• <b>±3 semitone 범위 선택:</b> ±3 semitone(약 ±19% 주파수 변화)은 실제 경적 음높이의 "
        "자연스러운 변동 범위 내에 있습니다. ±5 이상은 원본 소리와 지나치게 달라져 "
        "클래스 레이블 신뢰도가 떨어질 수 있습니다.",
    ]
    for r in ps_reasons:
        story.append(Paragraph(r, ST["bullet"]))
    story.append(sp(2))

    # 3.3 Gain
    story.append(Paragraph("3.3 Gain (볼륨 조절)", ST["h2"]))
    story.append(hr(C_BLUE, 0.4))

    story.append(Paragraph(
        "Gain은 오디오 신호의 <b>진폭(amplitude)을 배수로 곱하여 볼륨을 조절</b>하는 가장 단순한 변환입니다. "
        "데시벨(dB) 단위로 지정하며, 변환식은 <b>amplitude × 10^(dB/20)</b> 입니다.",
        ST["body"]))
    story.append(sp(1))

    gain_data = [
        [Paragraph("태그", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("dB 값", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("진폭 배수", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("효과", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER))],
        [Paragraph("gain_m4", ST["small"]), Paragraph("-4 dB", ST["small"]),
         Paragraph("×0.631", ST["small"]), Paragraph("원본보다 37% 조용한 버전", ST["small"])],
        [Paragraph("gain_p4", ST["small"]), Paragraph("+4 dB", ST["small"]),
         Paragraph("×1.585", ST["small"]), Paragraph("원본보다 59% 큰 버전 (clip 처리)", ST["small"])],
    ]
    gain_col = [22*mm, 20*mm, 30*mm, CONTENT_W - 72*mm]
    gain_tbl = Table(gain_data, colWidths=gain_col)
    gain_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_ORANGE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_LORANG, colors.white]),
        ("BOX", (0,0), (-1,-1), 0.8, C_ORANGE),
        ("INNERGRID", (0,0), (-1,-1), 0.3, C_LORANG),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("ALIGN", (0,0), (1,-1), "CENTER"),
    ]))
    story.append(gain_tbl)
    story.append(sp(2))

    story.append(Paragraph("<b>사용 이유:</b>", ST["h3"]))
    gain_reasons = [
        "• <b>녹음 환경 다양성:</b> 실제 앱 사용 시 스마트폰 볼륨, 마이크 감도, "
        "소음원과의 거리에 따라 입력 오디오 크기가 크게 달라집니다. "
        "Gain 변환으로 이런 다양한 음량 조건에서도 강인한 모델을 훈련합니다.",
        "• <b>np.clip 처리:</b> +4dB 증폭 시 -1.0~+1.0 범위를 벗어나는 샘플이 생길 수 있어 "
        "클리핑(clipping) 처리를 적용합니다. 이는 실제 마이크 포화(saturation) 현상을 시뮬레이션합니다.",
        "• <b>±4 dB 범위 선택:</b> YAMNet은 입력 전 정규화를 수행하므로 "
        "지나친 gain(±10dB 이상)은 효과가 없습니다. ±4dB는 전처리 후에도 "
        "특징 추출에 영향을 줄 수 있는 적절한 범위입니다.",
    ]
    for r in gain_reasons:
        story.append(Paragraph(r, ST["bullet"]))
    story.append(sp(2))

    # 3.4 Combo
    story.append(Paragraph("3.4 Combo (복합 변환)", ST["h2"]))
    story.append(hr(C_BLUE, 0.4))

    story.append(Paragraph(
        "Combo는 Time Stretch와 Pitch Shift를 <b>순차적으로 조합</b>하여 적용하는 변환입니다. "
        "단순히 두 변환의 합이 아니라, 시간-주파수 공간에서 서로 다른 방향으로 변형이 일어나 "
        "단일 변환으로는 만들 수 없는 새로운 오디오 특성을 생성합니다.",
        ST["body"]))
    story.append(sp(1))

    combo_data = [
        [Paragraph("태그", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("변환 조합", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("효과", ParagraphStyle("h", fontName="MGB", fontSize=9.5, textColor=colors.white, alignment=TA_CENTER))],
        [Paragraph("combo1", ST["small"]),
         Paragraph("ts0.90 (10% 감속) → ps+2 (+2 semitone)", ST["small"]),
         Paragraph("느리고 약간 높은 음: 거리가 먼 차량의 경적 시뮬레이션", ST["small"])],
        [Paragraph("combo2", ST["small"]),
         Paragraph("ts1.10 (10% 가속) → ps-2 (-2 semitone)", ST["small"]),
         Paragraph("빠르고 약간 낮은 음: 가까이 지나치는 차량의 경적 시뮬레이션", ST["small"])],
    ]
    combo_col = [20*mm, 60*mm, CONTENT_W - 80*mm]
    combo_tbl = Table(combo_data, colWidths=combo_col)
    combo_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_PURPLE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_LPURP, colors.white]),
        ("BOX", (0,0), (-1,-1), 0.8, C_PURPLE),
        ("INNERGRID", (0,0), (-1,-1), 0.3, C_LPURP),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("ALIGN", (0,0), (0,-1), "CENTER"),
    ]))
    story.append(combo_tbl)
    story.append(sp(2))

    story.append(Paragraph("<b>사용 이유:</b>", ST["h3"]))
    combo_reasons = [
        "• <b>도플러 효과 근사:</b> 자동차가 접근하거나 멀어질 때 도플러 효과로 인해 "
        "음높이와 음량이 동시에 변합니다. Combo 변환은 이를 간접적으로 시뮬레이션합니다.",
        "• <b>데이터 다양성 극대화:</b> 단일 변환만으로는 표현할 수 없는 복합적인 "
        "오디오 특성을 생성하여 모델이 더 다양한 패턴을 학습하도록 합니다.",
        "• <b>순서 중요성:</b> Time Stretch를 먼저 적용한 후 Pitch Shift를 적용합니다. "
        "반대 순서도 가능하지만, 먼저 시간 축을 정렬한 후 주파수를 조정하는 것이 "
        "위상 보코더 알고리즘의 안정성 측면에서 더 일관된 결과를 냅니다.",
    ]
    for r in combo_reasons:
        story.append(Paragraph(r, ST["bullet"]))

    story.append(PageBreak())

    # ── 4. SNR 노이즈 추가 ────────────────────────────
    story += section_header("4", "SNR 노이즈 추가 — 소음 환경 시뮬레이션")

    story.append(Paragraph(
        "증강된 clean 파일은 4가지 SNR(Signal-to-Noise Ratio) 조건에서도 동일하게 생성됩니다. "
        "이는 실제 도로 환경의 배경 소음을 시뮬레이션하기 위한 것입니다.",
        ST["body"]))
    story.append(sp(2))

    snr_data = [
        [Paragraph("SNR 조건", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("의미", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("실제 환경 예시", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("신호/소음 비율", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER))],
        [Paragraph("clean", ST["body_left"]), Paragraph("소음 없음 (원본)", ST["body_left"]),
         Paragraph("조용한 실내 녹음", ST["body_left"]), Paragraph("∞ dB", ST["body_left"])],
        [Paragraph("snr_+10dB", ST["body_left"]), Paragraph("신호가 소음보다 10dB 큼", ST["body_left"]),
         Paragraph("한적한 도로", ST["body_left"]), Paragraph("신호 전력 10배", ST["body_left"])],
        [Paragraph("snr_+5dB", ST["body_left"]), Paragraph("신호가 소음보다 5dB 큼", ST["body_left"]),
         Paragraph("일반 도심 도로", ST["body_left"]), Paragraph("신호 전력 3.16배", ST["body_left"])],
        [Paragraph("snr_+0dB", ST["body_left"]), Paragraph("신호와 소음이 동등", ST["body_left"]),
         Paragraph("혼잡한 교차로", ST["body_left"]), Paragraph("신호 전력 = 소음 전력", ST["body_left"])],
        [Paragraph("snr_-5dB", ST["body_left"]), Paragraph("소음이 신호보다 5dB 큼", ST["body_left"]),
         Paragraph("고속도로, 악천후", ST["body_left"]), Paragraph("소음 전력 3.16배", ST["body_left"])],
    ]
    snr_col = [26*mm, 40*mm, 36*mm, CONTENT_W - 102*mm]
    snr_tbl = Table(snr_data, colWidths=snr_col)
    snr_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_TEAL),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_LTEAL, colors.white, C_LTEAL, colors.white, C_LTEAL]),
        ("BOX", (0,0), (-1,-1), 0.8, C_TEAL),
        ("INNERGRID", (0,0), (-1,-1), 0.3, C_LTEAL),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(snr_tbl)
    story.append(sp(2))

    story.append(Paragraph(
        "SNR 노이즈 추가 공식: <b>y_noisy = signal + scale × noise</b><br/>"
        "여기서 scale = sqrt(P_signal / (P_noise × 10^(SNR/10)))",
        ST["body_left"]))

    story.append(PageBreak())

    # ── 5. 구현 코드 설명 ─────────────────────────────
    story += section_header("5", "구현 코드 상세 설명")

    # 5.1 환경 설정
    story.append(Paragraph("5.1 라이브러리 및 상수 설정", ST["h2"]))
    story += code_block("""\
import os, random, numpy as np, librosa, soundfile as sf

random.seed(42)      # 재현 가능한 결과를 위한 시드 고정
np.random.seed(42)

SR = 22050           # UrbanSound8K 표준 샘플링 레이트
DURATION = 4.0       # 표준 클립 길이 (초)
N_SAMPLES = int(SR * DURATION)   # = 88,200 샘플

PROCESSED = '/mnt/c/Users/.../data/processed'
SNR_TAGS  = ['snr_+10dB', 'snr_+5dB', 'snr_+0dB', 'snr_-5dB']
SNR_DB    = [10, 5, 0, -5]""")

    story.append(Paragraph(
        "librosa는 오디오 분석/변환 라이브러리이고, soundfile은 WAV 파일 읽기/쓰기를 담당합니다. "
        "SR=22050은 UrbanSound8K의 표준 샘플레이트이며, 모든 파일을 이 레이트로 통일합니다.",
        ST["body"]))
    story.append(sp(2))

    # 5.2 오디오 로드
    story.append(Paragraph("5.2 오디오 로드 및 길이 정규화", ST["h2"]))
    story += code_block("""\
def load_fixed(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))   # 짧으면 뒤에 0 패딩
    else:
        y = y[:N_SAMPLES]                          # 길면 앞부터 4초만 사용
    return y.astype(np.float32)""")

    story.append(Paragraph(
        "모든 오디오를 정확히 88,200 샘플(4초)로 맞춥니다. "
        "Time Stretch로 늘어난 파일은 앞부분만 사용하고, "
        "가속으로 짧아진 파일은 뒤에 무음(zero)을 채웁니다.",
        ST["body"]))
    story.append(sp(2))

    # 5.3 SNR 노이즈 추가
    story.append(Paragraph("5.3 SNR 노이즈 추가 함수", ST["h2"]))
    story += code_block("""\
def add_snr_noise(signal, snr_db):
    sig_power   = np.mean(signal ** 2) + 1e-10   # 신호 전력 (0 나눗셈 방지)
    noise       = np.random.randn(len(signal)).astype(np.float32)
    noise_power = np.mean(noise ** 2) + 1e-10    # 노이즈 전력
    scale = np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10)))
    return np.clip(signal + scale * noise, -1.0, 1.0)""")

    story.append(Paragraph(
        "가우시안 화이트 노이즈를 SNR 공식에 맞는 스케일로 조정하여 신호에 더합니다. "
        "np.clip으로 ±1.0 범위를 초과하지 않도록 처리합니다.",
        ST["body"]))
    story.append(sp(2))

    # 5.4 증강 풀
    story.append(Paragraph("5.4 증강 변환 풀 (12가지)", ST["h2"]))
    story += code_block("""\
AUG_POOL = [
    ('ts085',   lambda y: librosa.effects.time_stretch(y, rate=0.85)),
    ('ts090',   lambda y: librosa.effects.time_stretch(y, rate=0.90)),
    ('ts110',   lambda y: librosa.effects.time_stretch(y, rate=1.10)),
    ('ts115',   lambda y: librosa.effects.time_stretch(y, rate=1.15)),
    ('ps_m3',   lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=-3)),
    ('ps_m2',   lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=-2)),
    ('ps_p2',   lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=+2)),
    ('ps_p3',   lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=+3)),
    ('gain_m4', lambda y: np.clip(y * 10**(-4/20), -1, 1)),
    ('gain_p4', lambda y: np.clip(y * 10**(+4/20), -1, 1)),
    ('combo1',  lambda y: librosa.effects.pitch_shift(
                    librosa.effects.time_stretch(y, rate=0.90), sr=SR, n_steps=+2)),
    ('combo2',  lambda y: librosa.effects.pitch_shift(
                    librosa.effects.time_stretch(y, rate=1.10), sr=SR, n_steps=-2)),
]""")

    story.append(Paragraph(
        "(태그, 변환함수) 튜플의 리스트로 구성합니다. "
        "태그는 생성된 파일명에 포함되어 어떤 변환이 적용되었는지 추적할 수 있게 합니다. "
        "예: aug_162540-1-0-0_ts085_003.wav",
        ST["body"]))
    story.append(sp(2))

    # 5.5 메인 루프
    story.append(Paragraph("5.5 fold별 증강 메인 루프", ST["h2"]))
    story += code_block("""\
for fold in folds:
    # 1. siren과 horn 수 확인
    n_siren    = len([f for f in os.listdir(siren_dir) if f.endswith('.wav')])
    horn_files = [f for f in os.listdir(horn_dir) if f.endswith('.wav')]
    need       = n_siren - len(horn_files)   # 부족한 수 계산

    if need <= 0:
        continue   # fold5처럼 이미 충분하면 건너뜀

    # 2. (source 파일, aug 태그) 쌍 결정 — 순환 방식
    assignments = []
    for i in range(need):
        src = horn_files[i % len(horn_files)]    # 원본 순환
        tag = aug_tags[i % len(aug_tags)]        # 변환 순환
        assignments.append((src, tag))

    # 3. 변환 적용 및 저장
    for idx, (src_name, aug_tag) in enumerate(assignments):
        y     = load_fixed(os.path.join(horn_dir, src_name))
        y_aug = apply_aug(y, aug_tag)            # clean 변환
        new_name = f"aug_{stem}_{aug_tag}_{idx:03d}.wav"

        sf.write(os.path.join(horn_dir, new_name), y_aug, SR)   # clean 저장

        for snr_tag, snr_db in zip(SNR_TAGS, SNR_DB):           # SNR 4종 저장
            y_noisy = add_snr_noise(y_aug, snr_db)
            sf.write(os.path.join(snr_horn_dir, new_name), y_noisy, SR)""")

    story.append(Paragraph(
        "원본 파일과 변환 태그를 각각 순환(cycle)하여 배정합니다. "
        "예를 들어 fold1에서 36개 원본으로 50개를 만들어야 할 때, "
        "처음 36번은 각 원본을 1번씩 사용하고, 37~50번은 다시 처음부터 순환합니다. "
        "clean 저장 후 4개 SNR 조건에도 동일 파일을 노이즈 추가하여 저장합니다.",
        ST["body"]))

    story.append(PageBreak())

    # ── 6. 증강 결과 요약 ─────────────────────────────
    story += section_header("6", "증강 결과 요약")

    result_data = [
        [Paragraph("Fold", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("역할", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("증강 전 horn", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("증강 후 horn", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("siren", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("생성 수", ParagraphStyle("h", fontName="MGB", fontSize=10, textColor=colors.white, alignment=TA_CENTER))],
        ["fold1",  "Train", "36",  "86",  "86",  "50"],
        ["fold2",  "Train", "42",  "91",  "91",  "49"],
        ["fold3",  "Train", "43",  "119", "119", "76"],
        ["fold4",  "Train", "59",  "166", "166", "107"],
        ["fold5",  "Train", "98",  "98 (skip)", "71",  "0"],
        ["fold6",  "Train", "28",  "74",  "74",  "46"],
        ["fold7",  "Train", "28",  "77",  "77",  "49"],
        ["fold8",  "Train", "30",  "80",  "80",  "50"],
        ["fold9",  "Val",   "32",  "82",  "82",  "50"],
        ["fold10", "Test",  "33",  "83",  "83",  "50"],
        [Paragraph("<b>합계</b>", ST["body_left"]), "",
         Paragraph("<b>429</b>", ST["body_left"]),
         Paragraph("<b>956</b>", ST["body_left"]),
         Paragraph("<b>929</b>", ST["body_left"]),
         Paragraph("<b>527</b>", ST["body_left"])],
    ]
    res_col = [18*mm, 16*mm, 28*mm, 28*mm, 20*mm, 20*mm]
    # Convert plain strings to Paragraphs
    formatted_data = [result_data[0]]
    for row in result_data[1:-1]:
        role = row[1]
        role_color = {"Train": C_LGREEN, "Val": colors.HexColor("#FFF9C4"),
                      "Test": colors.HexColor("#FCE4EC")}.get(role, colors.white)
        formatted_data.append([Paragraph(str(v), ST["small"]) for v in row])
    formatted_data.append(result_data[-1] + [""] * (6 - len(result_data[-1])) if len(result_data[-1]) < 6 else result_data[-1])

    res_tbl = Table(result_data, colWidths=res_col)
    res_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_NAVY),
        ("BACKGROUND", (0,5), (-1,5), C_LGREEN),  # fold5 skip
        ("BACKGROUND", (0,-1), (-1,-1), C_LBLUE),  # 합계
        ("ROWBACKGROUNDS", (0,1), (-1,-2), [colors.white, C_LGRAY]),
        ("BOX", (0,0), (-1,-1), 1, C_BLUE),
        ("INNERGRID", (0,0), (-1,-1), 0.3, C_LBLUE),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME", (0,1), (-1,-1), "MG"),
        ("FONTSIZE", (0,1), (-1,-1), 9.5),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(res_tbl)
    story.append(sp(3))

    story += info_box(
        "<b>파일 생성 총계:</b><br/>"
        "• clean 증강 파일: 527개<br/>"
        "• SNR 조건별 (×4): 2,108개<br/>"
        "• <b>총 생성 파일: 2,635개</b><br/><br/>"
        "증강 후 car_horn 총계(clean): 429 + 527 = <b>956개</b>로 siren(929개)과 균형을 맞췄습니다. "
        "(fold5의 horn 98개 > siren 71개이나 fold5는 증강 없이 유지)",
        C_LLBLUE, C_BLUE
    )

    return story


# ── 빌드 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    out = "/mnt/c/Users/Daniel Park/Desktop/SoundPJ-rebuilt/reports/horn_augmentation_report.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    doc = SimpleDocTemplate(
        out,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=22*mm,
    )
    story = build_story()
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"PDF saved: {out}")
