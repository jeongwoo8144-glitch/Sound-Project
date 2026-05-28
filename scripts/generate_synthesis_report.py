"""
background + noise 합성 기법 설명 PDF 생성
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

pdfmetrics.registerFont(TTFont("MG",  "/mnt/c/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("MGB", "/mnt/c/Windows/Fonts/malgunbd.ttf"))

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
C_CODE_BG= colors.HexColor("#F8F9FA")
C_CODE_BD= colors.HexColor("#DEE2E6")
C_BROWN  = colors.HexColor("#784212")
C_LBROWN = colors.HexColor("#FAE5D3")

W, H    = A4
MARGIN  = 20 * mm
CW      = W - 2 * MARGIN

def S(name, **kw):
    base = {
        "title":    dict(fontName="MGB", fontSize=22, textColor=C_NAVY, spaceAfter=4, alignment=TA_CENTER),
        "subtitle": dict(fontName="MG",  fontSize=12, textColor=C_BLUE, spaceAfter=16, alignment=TA_CENTER),
        "h1":       dict(fontName="MGB", fontSize=15, textColor=C_NAVY, spaceBefore=18, spaceAfter=6, leading=20),
        "h2":       dict(fontName="MGB", fontSize=12, textColor=C_BLUE, spaceBefore=12, spaceAfter=4, leading=16),
        "h3":       dict(fontName="MGB", fontSize=10.5, textColor=C_GRAY, spaceBefore=8, spaceAfter=3, leading=14),
        "body":     dict(fontName="MG",  fontSize=10, textColor=C_DARK, leading=16, spaceAfter=4, alignment=TA_JUSTIFY),
        "bl":       dict(fontName="MG",  fontSize=10, textColor=C_DARK, leading=16, spaceAfter=4, alignment=TA_LEFT),
        "bullet":   dict(fontName="MG",  fontSize=10, textColor=C_DARK, leading=16, spaceAfter=3, leftIndent=12, firstLineIndent=-12),
        "code":     dict(fontName="MG",  fontSize=8.5, textColor=C_DARK, leading=13, spaceAfter=0),
        "small":    dict(fontName="MG",  fontSize=9,   textColor=C_DARK, leading=13, spaceAfter=2),
        "caption":  dict(fontName="MG",  fontSize=8.5, textColor=C_GRAY, spaceAfter=6, alignment=TA_CENTER),
        "tag":      dict(fontName="MGB", fontSize=9,   textColor=colors.white, spaceAfter=0, alignment=TA_CENTER),
    }
    params = base.get(name, {})
    params.update(kw)
    return ParagraphStyle(name, **params)

ST = {k: S(k) for k in ["title","subtitle","h1","h2","h3","body","bl","bullet","code","small","caption","tag"]}

def hr(c=C_BLUE, t=0.8):
    return HRFlowable(width="100%", thickness=t, color=c, spaceAfter=6, spaceBefore=2)

def sp(h=4):
    return Spacer(1, h*mm)

def code_block(raw, lang="Python"):
    lines = raw.strip("\n").split("\n")
    parts = []
    for line in lines:
        s = line.lstrip(" ")
        n = len(line) - len(s)
        ind = "&#160;" * n
        esc = s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        parts.append((ind + esc) if esc else "&#160;")
    inner = "<br/>".join(parts)

    lbl = Table([[Paragraph(lang, ST["tag"])]], colWidths=[CW])
    lbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), C_BLUE),
        ("TOPPADDING",(0,0),(-1,-1), 3), ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING",(0,0),(-1,-1), 8),
    ]))
    cod = Table([[Paragraph(inner, ST["code"])]], colWidths=[CW])
    cod.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), C_CODE_BG),
        ("BOX",(0,0),(-1,-1), 0.8, C_CODE_BD),
        ("TOPPADDING",(0,0),(-1,-1), 8), ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING",(0,0),(-1,-1), 10), ("RIGHTPADDING",(0,0),(-1,-1), 10),
    ]))
    return [lbl, cod, sp(2)]

def sec_hdr(num, title, c=C_NAVY):
    data = [[Paragraph(f"{num}. {title}", ParagraphStyle("x", fontName="MGB", fontSize=13,
        textColor=colors.white, leading=17, alignment=TA_LEFT))]]
    t = Table(data, colWidths=[CW])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), c),
        ("TOPPADDING",(0,0),(-1,-1), 7), ("BOTTOMPADDING",(0,0),(-1,-1), 7),
        ("LEFTPADDING",(0,0),(-1,-1), 12),
    ]))
    return [sp(2), t, sp(3)]

def info_box(text, bg=C_LLBLUE, bd=C_BLUE):
    t = Table([[Paragraph(text, ST["bl"])]], colWidths=[CW])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), bg),
        ("BOX",(0,0),(-1,-1), 1.2, bd),
        ("TOPPADDING",(0,0),(-1,-1), 8), ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING",(0,0),(-1,-1), 12), ("RIGHTPADDING",(0,0),(-1,-1), 12),
    ]))
    return [t, sp(2)]

def hdr_cell(text, fs=10):
    return Paragraph(text, ParagraphStyle("hc", fontName="MGB", fontSize=fs,
        textColor=colors.white, alignment=TA_CENTER, leading=14))

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("MG", 8)
    canvas.setFillColor(C_GRAY)
    canvas.drawRightString(W-MARGIN, 12*mm, f"{doc.page}")
    canvas.drawString(MARGIN, 12*mm, "ADAS Sound Detector — background 합성 데이터 생성 기법 보고서")
    canvas.setStrokeColor(C_LBLUE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 14*mm, W-MARGIN, 14*mm)
    canvas.restoreState()


def build():
    story = []

    # ── 표지 ─────────────────────────────────────────────
    story.append(sp(20))
    story.append(Paragraph("ADAS Sound Detector", ST["subtitle"]))
    story.append(Paragraph("background 합성 데이터 생성 기법 보고서", ST["title"]))
    story.append(sp(3))
    story.append(hr(C_BLUE, 1.5))
    story.append(sp(2))
    story.append(Paragraph("background + noise 혼합 합성을 통한 10,000개 학습 데이터 확장", ST["subtitle"]))
    story.append(sp(8))

    summ = [
        [hdr_cell("항목"), hdr_cell("내용")],
        [Paragraph("<b>생성 목적</b>", ST["bl"]), Paragraph("background 클래스 샘플 수 확대 및 실제 도로 소음 환경 다양성 반영", ST["small"])],
        [Paragraph("<b>소스 데이터</b>", ST["bl"]), Paragraph("data/processed/clean/fold*/background/ (2,000개) + data/noise/ (TCAR 16, engine 663, road 623)", ST["small"])],
        [Paragraph("<b>생성 수량</b>", ST["bl"]), Paragraph("총 10,000개 (fold당 1,000개 균등 분배)", ST["small"])],
        [Paragraph("<b>원본 보존</b>", ST["bl"]), Paragraph("기존 파일 수정 없음. 신규 파일명 synth_foldN_[cat]_[SNR]dB_XXXX.wav 형식", ST["small"])],
        [Paragraph("<b>핵심 기법</b>", ST["bl"]), Paragraph("SNR Mixing / Random Crop / RMS 정규화 / Random Gain / 노이즈 카테고리 균형 혼합", ST["small"])],
    ]
    st = Table(summ, colWidths=[35*mm, CW-35*mm])
    st.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), C_NAVY),
        ("BACKGROUND",(0,1),(-1,-1), colors.white),
        ("BACKGROUND",(0,1),(0,-1), C_LBLUE),
        ("BOX",(0,0),(-1,-1), 1.2, C_BLUE),
        ("INNERGRID",(0,0),(-1,-1), 0.3, C_LBLUE),
        ("VALIGN",(0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1), 7), ("BOTTOMPADDING",(0,0),(-1,-1), 7),
        ("LEFTPADDING",(0,0),(-1,-1), 10),
    ]))
    story.append(st)
    story.append(PageBreak())

    # ── 1. 합성이 필요한 이유 ─────────────────────────────
    story += sec_hdr("1", "합성이 필요한 이유")

    story.append(Paragraph(
        "ADAS 소리 감지 앱은 실제 도로 주행 환경에서 사용됩니다. "
        "그러나 UrbanSound8K의 background 클래스(자동차 엔진음, 도로 소음 등)는 "
        "조용한 환경에서 수집된 비교적 깨끗한 오디오입니다. "
        "실제 차량 내부에서는 background 소리와 다양한 주변 노이즈가 동시에 존재합니다.", ST["body"]))

    story.append(Paragraph(
        "또한 background 클래스는 현재 2,000개이며, "
        "car_horn(956개)이나 siren(929개)에 비해 과도하게 많아 "
        "모델이 background에 과적합될 위험이 있습니다. "
        "합성 데이터를 추가하여 <b>실제 환경 재현</b>과 <b>데이터 다양성 향상</b>을 동시에 달성합니다.", ST["body"]))
    story.append(sp(2))

    noise_data = [
        [hdr_cell("노이즈 소스"), hdr_cell("파일 수"), hdr_cell("대표 환경"), hdr_cell("파일당 길이")],
        ["TCAR\n(Traffic Car)", "16개", "고속도로/간선도로\n연속 주행 소음", "약 54초 (9.4MB)"],
        ["engine", "663개", "차량 엔진 아이들링\n및 주행 엔진음", "약 4초 (689KB)"],
        ["road", "623개", "도로면 마찰음\n타이어 구름 소리", "약 4초 (689KB)"],
    ]
    noise_tbl = Table(noise_data, colWidths=[30*mm, 22*mm, 48*mm, CW-100*mm])
    noise_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), C_TEAL),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_LTEAL, colors.white, C_LTEAL]),
        ("BOX",(0,0),(-1,-1), 0.8, C_TEAL),
        ("INNERGRID",(0,0),(-1,-1), 0.3, C_LTEAL),
        ("FONTNAME",(0,1),(-1,-1), "MG"), ("FONTSIZE",(0,1),(-1,-1), 9.5),
        ("VALIGN",(0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1), 6), ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING",(0,0),(-1,-1), 8), ("ALIGN",(1,0),(1,-1), "CENTER"),
    ]))
    story.append(noise_tbl)
    story.append(sp(2))

    story += info_box(
        "<b>노이즈 카테고리 가중치:</b> TCAR:engine:road = 2:2:1<br/>"
        "TCAR와 engine은 운전 중 가장 자주 경험하는 소음이므로 높은 가중치를 부여합니다. "
        "road 마찰음은 상대적으로 낮은 주파수 대역이라 background와 혼동될 수 있으므로 적은 비율로 사용합니다.",
        C_LTEAL, C_TEAL
    )

    # ── 2. 합성 파이프라인 개요 ───────────────────────────
    story += sec_hdr("2", "합성 파이프라인 개요")

    story.append(Paragraph(
        "각 합성 샘플은 아래 5단계 파이프라인을 거쳐 생성됩니다:", ST["body"]))
    story.append(sp(2))

    pipeline_data = [
        [hdr_cell("단계"), hdr_cell("기법"), hdr_cell("입력"), hdr_cell("출력")],
        ["①", "파일 선택\n(Random Selection)", "background 파일 pool\n+ noise 파일 pool", "src_bg, src_noi\n (카테고리 가중치 적용)"],
        ["②", "Random Crop\n(무작위 구간 추출)", "긴 오디오 파일\n(특히 TCAR ~54초)", "4초 클립\n(무작위 시작점)"],
        ["③", "SNR Mixing\n(신호 대 잡음비 혼합)", "bg 클립 + noise 클립\nSNR: -5 ~ +15 dB", "혼합 오디오\n(9가지 SNR 중 랜덤)"],
        ["④", "RMS 정규화\n(Gain Normalization)", "혼합 오디오", "-20 dBFS 기준\n정규화된 오디오"],
        ["⑤", "Random Gain\n(무작위 볼륨)", "정규화된 오디오", "±4 dB 변동 적용\n최종 출력 WAV"],
    ]
    pip_col = [12*mm, 32*mm, 48*mm, CW-92*mm]
    pip_tbl = Table(pipeline_data, colWidths=pip_col)
    pip_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), C_NAVY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_LLBLUE, colors.white]*3),
        ("BOX",(0,0),(-1,-1), 0.8, C_BLUE),
        ("INNERGRID",(0,0),(-1,-1), 0.3, C_LBLUE),
        ("FONTNAME",(0,1),(-1,-1), "MG"), ("FONTSIZE",(0,1),(-1,-1), 9),
        ("VALIGN",(0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1), 6), ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING",(0,0),(-1,-1), 6),
        ("ALIGN",(0,0),(0,-1), "CENTER"),
    ]))
    story.append(pip_tbl)
    story.append(PageBreak())

    # ── 3. 기법별 상세 ───────────────────────────────────
    story += sec_hdr("3", "기법별 상세 설명")

    # 3.1 SNR Mixing
    story.append(Paragraph("3.1 SNR Mixing (신호 대 잡음비 기반 혼합)", ST["h2"]))
    story.append(hr(C_BLUE, 0.4))
    story.append(Paragraph(
        "SNR Mixing은 배경 소리(background)와 노이즈를 <b>지정된 신호 대 잡음비(SNR)</b>로 혼합하는 기법입니다. "
        "SNR이 높을수록 background가 선명하게 들리고, 낮을수록 노이즈가 지배적이 됩니다. "
        "혼합 공식은 다음과 같습니다:", ST["body"]))
    story.append(sp(1))
    story += code_block("""\
def mix_snr(signal, noise, snr_db):
    sig_p  = np.mean(signal ** 2) + 1e-10   # 신호 전력 계산
    noi_p  = np.mean(noise  ** 2) + 1e-10   # 노이즈 전력 계산
    # SNR 공식: scale = sqrt(P_sig / (P_noi * 10^(SNR/10)))
    scale  = np.sqrt(sig_p / (noi_p * 10 ** (snr_db / 10)))
    mixed  = signal + scale * noise
    return normalize_rms(mixed)              # RMS 정규화 적용""")

    story.append(Paragraph(
        "9가지 SNR 값(-5, -3, 0, +3, +5, +8, +10, +12, +15 dB)을 균등 확률로 선택합니다. "
        "이 범위는 실제 차량 내 마이크로 측정한 도로 소음 환경(일반 주행 ~0~+10 dB, "
        "고속도로 ~-5~+5 dB, 조용한 도로 ~+10~+15 dB)을 포괄합니다.", ST["body"]))
    story.append(sp(1))

    snr_data = [
        [hdr_cell("SNR 범위"), hdr_cell("실제 환경"), hdr_cell("background 명료도"), hdr_cell("학습 효과")],
        ["-5 ~ 0 dB", "고속도로\n악천후", "노이즈 지배적", "극한 환경 강인성"],
        ["+3 ~ +8 dB", "도심 혼잡\n일반 주행", "background와 노이즈 균형", "현실 환경 재현"],
        ["+10 ~ +15 dB", "한적한 도로\n저속 주행", "background 선명", "깨끗한 환경 학습"],
    ]
    snr_tbl = Table(snr_data, colWidths=[28*mm, 36*mm, 38*mm, CW-102*mm])
    snr_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), C_BLUE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_LRED, C_LGREEN, C_LLBLUE]),
        ("BOX",(0,0),(-1,-1), 0.8, C_BLUE),
        ("INNERGRID",(0,0),(-1,-1), 0.3, C_LBLUE),
        ("FONTNAME",(0,1),(-1,-1), "MG"), ("FONTSIZE",(0,1),(-1,-1), 9.5),
        ("VALIGN",(0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1), 6), ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING",(0,0),(-1,-1), 8), ("ALIGN",(0,0),(0,-1), "CENTER"),
    ]))
    story.append(snr_tbl)
    story.append(sp(2))

    story.append(Paragraph("<b>사용 이유:</b>", ST["h3"]))
    for r in [
        "• <b>현실 환경 재현:</b> 실제 앱 사용 시 마이크는 엔진 소음, 도로 마찰음, 바람 소리가 섞인 "
        "환경에서 동작합니다. SNR 혼합으로 이런 다양한 소음 조건을 시뮬레이션합니다.",
        "• <b>False Negative 감소:</b> 모델이 소음 환경에서도 background를 정확히 인식하도록 "
        "다양한 SNR 조건의 학습 데이터를 제공합니다.",
        "• <b>정밀한 비율 제어:</b> 단순 덧셈 대신 전력 기반 스케일링으로 원하는 SNR을 "
        "정확하게 달성합니다.",
    ]:
        story.append(Paragraph(r, ST["bullet"]))
    story.append(sp(2))

    # 3.2 Random Crop
    story.append(Paragraph("3.2 Random Crop (무작위 구간 추출)", ST["h2"]))
    story.append(hr(C_BLUE, 0.4))
    story.append(Paragraph(
        "Random Crop은 <b>4초보다 긴 오디오 파일에서 무작위 시작점을 선택해 4초 구간을 추출</b>하는 기법입니다. "
        "TCAR 파일처럼 약 54초짜리 긴 파일에서 항상 앞부분만 사용하면 데이터 다양성이 낮아집니다.", ST["body"]))
    story.append(sp(1))
    story += code_block("""\
def load_fixed(path, sr=SR):
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if len(y) < N_SAMP:                           # 4초보다 짧으면
        y = np.pad(y, (0, N_SAMP - len(y)))       # 끝에 zero padding
    else:                                          # 4초보다 길면
        max_start = len(y) - N_SAMP
        start = random.randint(0, max_start)       # 무작위 시작점 선택
        y = y[start : start + N_SAMP]             # 4초 추출
    return y.astype(np.float32)""")

    story.append(Paragraph("<b>사용 이유:</b>", ST["h3"]))
    for r in [
        "• <b>TCAR 파일 최대 활용:</b> 54초짜리 TCAR 파일에서 이론적으로 최대 13가지 다른 구간을 "
        "추출할 수 있습니다. 무작위 추출로 16개 파일로부터 다양한 교통 소음 패턴을 생성합니다.",
        "• <b>시간적 다양성:</b> 같은 소음 파일도 시간에 따라 소리 특성이 달라집니다 "
        "(가속 구간, 정속 구간, 감속 구간 등). Random Crop으로 이런 변화를 자동으로 반영합니다.",
        "• <b>고정 길이 보장:</b> 모든 출력 파일을 4초(88,200 샘플)로 통일하여 "
        "YAMNet 입력 규격과 일치시킵니다.",
    ]:
        story.append(Paragraph(r, ST["bullet"]))
    story.append(sp(2))

    # 3.3 RMS Normalization
    story.append(Paragraph("3.3 RMS 정규화 (Gain Normalization)", ST["h2"]))
    story.append(hr(C_BLUE, 0.4))
    story.append(Paragraph(
        "RMS(Root Mean Square) 정규화는 오디오의 <b>평균 음량(RMS 에너지)을 목표 레벨로 맞추는</b> 기법입니다. "
        "혼합 후 음량 레벨이 불규칙해지는 문제를 해결합니다. "
        "목표 레벨은 방송/음성 처리 표준인 <b>-20 dBFS</b>로 설정했습니다.", ST["body"]))
    story.append(sp(1))
    story += code_block("""\
def rms_db(y):
    rms = np.sqrt(np.mean(y ** 2) + 1e-10)
    return 20 * np.log10(rms)         # dBFS 변환

def normalize_rms(y, target_db=-20.0):
    current = rms_db(y)               # 현재 RMS 레벨 측정
    gain = 10 ** ((target_db - current) / 20)   # 필요한 gain 계산
    return np.clip(y * gain, -1.0, 1.0)          # 적용 후 클리핑 방지""")

    story.append(Paragraph("<b>사용 이유:</b>", ST["h3"]))
    for r in [
        "• <b>일관된 음량 수준:</b> SNR 혼합 후 파일마다 음량이 크게 달라질 수 있습니다. "
        "YAMNet은 입력 진폭에 민감하므로 일관된 음량 수준을 유지해야 학습이 안정적입니다.",
        "• <b>-20 dBFS 기준:</b> 음성 처리 분야의 표준 레벨로, 클리핑(-0 dBFS)과 "
        "너무 조용한 신호(-40 dBFS 이하) 사이의 적절한 지점입니다.",
        "• <b>클리핑 방지:</b> np.clip으로 ±1.0 범위를 넘지 않도록 보호합니다.",
    ]:
        story.append(Paragraph(r, ST["bullet"]))
    story.append(sp(2))

    # 3.4 Random Gain
    story.append(Paragraph("3.4 Random Gain (무작위 볼륨 변동)", ST["h2"]))
    story.append(hr(C_BLUE, 0.4))
    story.append(Paragraph(
        "RMS 정규화 후 추가로 <b>±4 dB의 무작위 볼륨 변동</b>을 적용합니다. "
        "이는 스마트폰 모델별 마이크 감도 차이, 거치 위치 차이, "
        "앱 볼륨 설정 차이를 시뮬레이션합니다.", ST["body"]))
    story.append(sp(1))
    story += code_block("""\
def random_gain(y, max_db=4.0):
    db  = random.uniform(-max_db, max_db)   # -4 ~ +4 dB 무작위 선택
    return np.clip(y * 10 ** (db / 20), -1.0, 1.0)""")

    story.append(Paragraph("<b>사용 이유:</b>", ST["h3"]))
    for r in [
        "• <b>마이크 다양성 모사:</b> 삼성 갤럭시, 아이폰, LG 등 기기마다 마이크 감도가 "
        "최대 6~8 dB 차이 날 수 있습니다. ±4 dB 범위가 현실적인 변동 범위입니다.",
        "• <b>RMS 정규화와의 역할 분리:</b> RMS 정규화는 과도한 음량 편차를 잡는 1차 처리이고, "
        "Random Gain은 정규화 후에도 남아있는 현실적인 음량 다양성을 추가하는 2차 처리입니다.",
    ]:
        story.append(Paragraph(r, ST["bullet"]))
    story.append(sp(2))

    # 3.5 Noise Category Mix
    story.append(Paragraph("3.5 노이즈 카테고리 균형 혼합", ST["h2"]))
    story.append(hr(C_BLUE, 0.4))
    story.append(Paragraph(
        "세 가지 노이즈 카테고리(TCAR, engine, road)를 <b>가중치 기반으로 균형있게 선택</b>합니다. "
        "단순 랜덤 선택은 파일 수가 많은 engine이 압도적으로 많이 선택되는 편향이 생깁니다.", ST["body"]))
    story.append(sp(1))
    story += code_block("""\
NOISE_CATS  = ['TCAR', 'engine', 'road']
CAT_WEIGHTS = [2, 2, 1]          # TCAR:engine:road = 40%:40%:20%

# 카테고리 선택
cat     = random.choices(NOISE_CATS, weights=CAT_WEIGHTS, k=1)[0]
# 선택된 카테고리 내에서 파일 무작위 선택
src_noi = random.choice(noise_pools[cat])""")

    cat_data = [
        [hdr_cell("카테고리"), hdr_cell("원본 파일 수"), hdr_cell("선택 가중치"), hdr_cell("선택 비율"), hdr_cell("선택 이유")],
        ["TCAR",   "16개",  "2", "40%", "주행 중 가장 빈번한\n교통 소음"],
        ["engine", "663개", "2", "40%", "차량 내부 엔진음\n(앱 실사용 환경)"],
        ["road",   "623개", "1", "20%", "저주파 특성으로\nbg와 유사성 주의"],
    ]
    cat_tbl = Table(cat_data, colWidths=[20*mm, 24*mm, 24*mm, 22*mm, CW-90*mm])
    cat_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), C_BROWN),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_LBROWN, colors.white, C_LBROWN]),
        ("BOX",(0,0),(-1,-1), 0.8, C_BROWN),
        ("INNERGRID",(0,0),(-1,-1), 0.3, C_LBROWN),
        ("FONTNAME",(0,1),(-1,-1), "MG"), ("FONTSIZE",(0,1),(-1,-1), 9),
        ("VALIGN",(0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1), 6), ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING",(0,0),(-1,-1), 8), ("ALIGN",(1,0),(3,-1), "CENTER"),
    ]))
    story.append(cat_tbl)
    story.append(PageBreak())

    # ── 4. 구현 코드 전체 ─────────────────────────────────
    story += sec_hdr("4", "구현 코드 전체")

    story.append(Paragraph("4.1 라이브러리 및 상수 설정", ST["h2"]))
    story += code_block("""\
import os, random, numpy as np, librosa, soundfile as sf
from pathlib import Path

random.seed(7)         # 재현 가능한 결과를 위한 시드
np.random.seed(7)

SR       = 22050       # UrbanSound8K 표준 샘플레이트
DURATION = 4.0         # 클립 길이 (초)
N_SAMP   = int(SR * DURATION)   # = 88,200 샘플

TARGET   = 10_000      # 총 생성 수
PER_FOLD = TARGET // 10            # = fold당 1,000개

SNR_CHOICES = [-5, -3, 0, 3, 5, 8, 10, 12, 15]   # 9가지 SNR
NOISE_CATS  = ['TCAR', 'engine', 'road']
CAT_WEIGHTS = [2, 2, 1]""")

    story.append(Paragraph("4.2 파일 목록 수집", ST["h2"]))
    story += code_block("""\
# background 파일 수집 (전체 fold)
bg_files = []
for fold_dir in sorted((BASE_PROC / 'clean').iterdir()):
    bg_dir = fold_dir / 'background'
    if bg_dir.is_dir():
        bg_files += list(bg_dir.glob('*.wav'))

# noise 카테고리별 풀 구성
noise_pools = {}
for cat in NOISE_CATS:
    cat_dir = NOISE_DIR / cat
    if cat_dir.is_dir():
        noise_pools[cat] = list(cat_dir.glob('*.wav'))""")

    story.append(Paragraph("4.3 fold별 생성 메인 루프", ST["h2"]))
    story += code_block("""\
for fold_dir in folds:
    out_dir = fold_dir / 'background'

    for i in range(PER_FOLD):
        # [1] background 소스: 같은 fold 파일 우선, 없으면 전체 pool
        fold_bg = [f for f in bg_files if fold_dir.name in str(f)]
        src_bg  = random.choice(fold_bg if fold_bg else bg_files)

        # [2] 노이즈 카테고리 가중치 선택 후 파일 선택
        cat     = random.choices(NOISE_CATS, weights=CAT_WEIGHTS, k=1)[0]
        src_noi = random.choice(noise_pools[cat])

        # [3] SNR 무작위 선택
        snr = random.choice(SNR_CHOICES)

        # [4] 오디오 로드 (Random Crop 내장)
        y_bg  = load_fixed(src_bg)
        y_noi = load_fixed(src_noi)

        # [5] SNR Mixing → RMS 정규화
        y_mix = mix_snr(y_bg, y_noi, snr)

        # [6] Random Gain (±4dB)
        y_mix = random_gain(y_mix)

        # [7] 저장: synth_fold1_TCAR_+5dB_0042.wav 형식
        fname = f"synth_{fold_dir.name}_{cat}_{snr:+d}dB_{i:04d}.wav"
        sf.write(str(out_dir / fname), y_mix, SR)""")

    story.append(Paragraph(
        "파일명에 fold, 노이즈 카테고리, SNR 정보가 모두 인코딩되어 있어 "
        "나중에 어떤 조건으로 생성되었는지 파일명만으로 추적 가능합니다.",
        ST["body"]))

    story.append(PageBreak())

    # ── 5. 생성 결과 요약 ─────────────────────────────────
    story += sec_hdr("5", "생성 결과 요약")

    fold_data = [
        [hdr_cell("Fold"), hdr_cell("역할"), hdr_cell("기존 background"), hdr_cell("신규 합성"), hdr_cell("총계")],
        ["fold1",  "Train", "201", "1,000", "1,201"],
        ["fold2",  "Train", "202", "1,000", "1,202"],
        ["fold3",  "Train", "203", "1,000", "1,203"],
        ["fold4",  "Train", "203", "1,000", "1,203"],
        ["fold5",  "Train", "203", "1,000", "1,203"],
        ["fold6",  "Train", "198", "1,000", "1,198"],
        ["fold7",  "Train", "200", "1,000", "1,200"],
        ["fold8",  "Train", "196", "1,000", "1,196"],
        ["fold9",  "Val",   "196", "1,000", "1,196"],
        ["fold10", "Test",  "198", "1,000", "1,198"],
        [Paragraph("<b>합계</b>", ST["bl"]), "",
         Paragraph("<b>2,000</b>", ST["bl"]),
         Paragraph("<b>10,000</b>", ST["bl"]),
         Paragraph("<b>12,000</b>", ST["bl"])],
    ]
    fold_col = [18*mm, 16*mm, 32*mm, 28*mm, 26*mm]
    fold_tbl = Table(fold_data, colWidths=fold_col)
    fold_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), C_NAVY),
        ("ROWBACKGROUNDS",(0,1),(-1,-2), [colors.white, C_LGRAY]),
        ("BACKGROUND",(0,-1),(-1,-1), C_LBLUE),
        ("BOX",(0,0),(-1,-1), 1, C_BLUE),
        ("INNERGRID",(0,0),(-1,-1), 0.3, C_LBLUE),
        ("FONTNAME",(0,1),(-1,-1), "MG"), ("FONTSIZE",(0,1),(-1,-1), 9.5),
        ("ALIGN",(0,0),(-1,-1), "CENTER"),
        ("VALIGN",(0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1), 5), ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    story.append(fold_tbl)
    story.append(sp(3))

    story += info_box(
        "<b>클래스 균형 비교 (합성 후):</b><br/>"
        "• background: 12,000개 (기존 2,000 + 합성 10,000)<br/>"
        "• car_horn:    956개 (clean 기준, SNR 포함 4,780개)<br/>"
        "• siren:       929개 (clean 기준, SNR 포함 4,645개)<br/><br/>"
        "<b>주의:</b> background는 clean 조건에만 존재합니다. "
        "SNR 조건(snr_+10dB 등)의 background 폴더는 원래 없으며, "
        "이번 합성에서도 clean에만 추가하여 기존 데이터 구조를 유지합니다.",
        C_LORANG, C_ORANGE
    )

    story += info_box(
        "<b>원본 데이터 보호:</b><br/>"
        "• 기존 파일명 형식: 156194-0-0-0.wav (UrbanSound8K 원본 형식)<br/>"
        "• 신규 파일명 형식: synth_fold1_TCAR_+5dB_0042.wav<br/>"
        "• 두 형식이 명확히 구분되므로 나중에 합성 파일만 선택적으로 제거 가능합니다.",
        C_LGREEN, C_GREEN
    )

    return story


if __name__ == "__main__":
    out = "/mnt/c/Users/Daniel Park/Desktop/SoundPJ-rebuilt/reports/background_synthesis_report.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    doc = SimpleDocTemplate(out, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=22*mm)
    doc.build(build(), onFirstPage=on_page, onLaterPages=on_page)
    print(f"PDF saved: {out}")
