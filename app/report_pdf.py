"""PDF report generator for AI vs Real image classification results.

Responsibility: build and return PDF bytes from a results DataFrame.
Has no Streamlit dependencies.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import List

import pandas as pd
from reportlab.graphics import renderPDF  # noqa: F401
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from zoneinfo import ZoneInfo


# ── Color palette ─────────────────────────────────────────────────────────
DARK = colors.HexColor("#0f1117")
SURFACE = colors.HexColor("#1e2130")
ACCENT = colors.HexColor("#4f8ef7")
SUCCESS = colors.HexColor("#21c55d")
ERROR_COLOR = colors.HexColor("#ef4444")
WARNING = colors.HexColor("#f59e0b")
TEXT_LIGHT = colors.HexColor("#e2e8f0")
TEXT_MUTED = colors.HexColor("#94a3b8")
WHITE = colors.white

REPORT_TIMEZONE = ZoneInfo("America/Bogota")


# ── Styles ────────────────────────────────────────────────────────────────
def _styles() -> dict:
    """Build and return a dict of ReportLab ParagraphStyle objects.

    Returns:
        Dict mapping style name strings to ParagraphStyle instances.
    """
    return {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=26,
            textColor=WHITE,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=12,
            textColor=TEXT_MUTED,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "section": ParagraphStyle(
            "section",
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=ACCENT,
            spaceBefore=16,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9,
            textColor=TEXT_LIGHT,
            leading=14,
            spaceAfter=4,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer",
            fontName="Helvetica",
            fontSize=8,
            textColor=TEXT_MUTED,
            leading=13,
            spaceAfter=3,
        ),
        "stat_label": ParagraphStyle(
            "stat_label",
            fontName="Helvetica",
            fontSize=8,
            textColor=TEXT_MUTED,
            alignment=TA_CENTER,
        ),
        "stat_value": ParagraphStyle(
            "stat_value",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica",
            fontSize=7,
            textColor=TEXT_MUTED,
            alignment=TA_CENTER,
        ),
    }


# ── Dark page background ──────────────────────────────────────────────────
def _dark_background(canvas, doc):
    """Draw a dark background with an accent stripe and footer on each page.

    Args:
        canvas: ReportLab canvas object for the current page.
        doc: The SimpleDocTemplate document being built.
    """
    canvas.saveState()
    canvas.setFillColor(DARK)
    canvas.rect(0, 0, A4[0], A4[1], fill=True, stroke=False)

    # Accent stripe at the top
    canvas.setFillColor(ACCENT)
    canvas.rect(0, A4[1] - 4, A4[0], 4, fill=True, stroke=False)

    # Footer
    canvas.setFillColor(TEXT_MUTED)
    canvas.setFont("Helvetica", 7)
    date_str = datetime.now(REPORT_TIMEZONE).strftime("%Y-%m-%d %H:%M COT")
    footer_text = (
        f"AI vs Real Image Detector  |  "
        f"Reporte generado el {date_str}  |  "
        f"Pagina {doc.page}"
    )
    canvas.drawCentredString(A4[0] / 2, 1.0 * cm, footer_text)
    canvas.restoreState()


# ── Cover page ────────────────────────────────────────────────────────────
def _build_cover(styles: dict, df: pd.DataFrame) -> list:
    """Build the cover page story elements.

    Args:
        styles: Dict of ParagraphStyle objects.
        df: Results DataFrame used to compute cover statistics.

    Returns:
        List of ReportLab flowable elements for the cover page.
    """
    story = []
    story.append(Spacer(1, 3.5 * cm))

    story.append(Paragraph("AI vs Real", styles["title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            "Reporte de Clasificacion de Imagenes",
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    ts = datetime.now(REPORT_TIMEZONE).strftime("%d %B %Y  %H:%M COT")
    story.append(Paragraph(ts, styles["subtitle"]))
    story.append(Spacer(1, 0.8 * cm))

    story.append(
        HRFlowable(width="80%", thickness=1, color=ACCENT, spaceAfter=20)
    )

    # Mini summary on cover page
    total = len(df)
    exitosas = len(df[df["status"] == "ok"])
    fallidas = total - exitosas

    stats = [
        [
            Paragraph(str(total), styles["stat_value"]),
            Paragraph(str(exitosas), styles["stat_value"]),
            Paragraph(str(fallidas), styles["stat_value"]),
        ],
        [
            Paragraph("Total", styles["stat_label"]),
            Paragraph("Exito", styles["stat_label"]),
            Paragraph("Error", styles["stat_label"]),
        ],
    ]

    tbl = Table(stats, colWidths=[5 * cm, 5 * cm, 5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [SURFACE, SURFACE]),
        ("BOX", (0, 0), (-1, -1), 0.5, ACCENT),
        ("LINEAFTER", (0, 0), (1, -1), 0.5, ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tbl)
    story.append(PageBreak())
    return story


# ── Disclaimer ────────────────────────────────────────────────────────────
def _build_disclaimer(styles: dict) -> list:
    """Build the responsible use disclaimer story elements.

    Args:
        styles: Dict of ParagraphStyle objects.

    Returns:
        List of ReportLab flowable elements for the disclaimer section.
    """
    story = []
    story.append(
        Paragraph("Disclaimer de Uso Responsable", styles["section"])
    )
    story.append(
        HRFlowable(
            width="100%", thickness=0.5, color=SURFACE, spaceAfter=8
        )
    )

    items = [
        "Esta herramienta es <b>de apoyo</b> para verificacion preliminar.",
        "<b>No</b> constituye una certificacion <b>forense ni legal</b>.",
        (
            "Los resultados son <b>probabilisticos</b> y pueden contener "
            "errores; no se garantiza exactitud del 100%."
        ),
        "No debe usarse como unica base para decisiones criticas.",
    ]
    for item in items:
        story.append(Paragraph(f"&#x2022;  {item}", styles["disclaimer"]))

    story.append(Spacer(1, 0.4 * cm))
    return story


# ── Statistical summary ───────────────────────────────────────────────────
def _build_summary(styles: dict, df: pd.DataFrame) -> list:
    """Build the statistical summary section.

    Args:
        styles: Dict of ParagraphStyle objects.
        df: Results DataFrame.

    Returns:
        List of ReportLab flowable elements for the summary section.
    """
    story = []
    story.append(Paragraph("Resumen Estadistico", styles["section"]))
    story.append(
        HRFlowable(
            width="100%", thickness=0.5, color=SURFACE, spaceAfter=8
        )
    )

    total = len(df)
    exitosas = len(df[df["status"] == "ok"])
    fallidas = total - exitosas
    ai_count = len(df[df["predicted_label"] == "ai"])
    real_count = len(df[df["predicted_label"] == "real"])

    tasa = f"{(exitosas / total * 100):.1f}%" if total > 0 else "N/A"

    rows = [
        ["Metrica", "Valor"],
        ["Total de imagenes analizadas", str(total)],
        ["Procesadas con exito", str(exitosas)],
        ["Con error", str(fallidas)],
        ["Tasa de exito", tasa],
        ["Clasificadas como IA", str(ai_count)],
        ["Clasificadas como Real", str(real_count)],
    ]

    col_w = [10 * cm, 5 * cm]
    tbl = Table(rows, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [SURFACE, DARK]),
        ("TEXTCOLOR", (0, 1), (-1, -1), TEXT_LIGHT),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 0.5, ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 1, ACCENT),
        ("LINEAFTER", (0, 0), (0, -1), 0.5, SURFACE),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.6 * cm))

    # Pie chart if there are valid predictions
    if exitosas > 0 and (ai_count > 0 or real_count > 0):
        story.extend(_build_pie_chart(ai_count, real_count, styles))

    return story


# ── Pie chart ─────────────────────────────────────────────────────────────
def _build_pie_chart(ai_count: int, real_count: int, styles: dict) -> list:
    """Build the predictions distribution pie chart section.

    Args:
        ai_count: Number of images classified as AI-generated.
        real_count: Number of images classified as real.
        styles: Dict of ParagraphStyle objects.

    Returns:
        List of ReportLab flowable elements for the pie chart section.
    """
    story = []
    story.append(
        Paragraph("Distribucion de Predicciones", styles["section"])
    )
    story.append(
        HRFlowable(
            width="100%", thickness=0.5, color=SURFACE, spaceAfter=8
        )
    )

    drawing = Drawing(400, 180)

    pie_chart = Pie()
    pie_chart.x = 100
    pie_chart.y = 20
    pie_chart.width = 140
    pie_chart.height = 140
    pie_chart.data = (
        [ai_count, real_count] if real_count > 0 else [ai_count, 0.001]
    )
    pie_chart.labels = ["IA", "Real"]
    pie_chart.slices[0].fillColor = ACCENT
    pie_chart.slices[1].fillColor = SUCCESS
    pie_chart.slices[0].strokeColor = DARK
    pie_chart.slices[1].strokeColor = DARK
    pie_chart.slices[0].strokeWidth = 1
    pie_chart.slices[1].strokeWidth = 1
    pie_chart.slices[0].fontColor = colors.white
    pie_chart.slices[1].fontColor = colors.white
    pie_chart.sideLabels = False
    drawing.add(pie_chart)

    # Manual legend
    legend_x = 270
    legend_y = 130
    for i, (label, clr, count) in enumerate([
        ("IA", ACCENT, ai_count),
        ("Real", SUCCESS, real_count),
    ]):
        rect = Rect(
            legend_x,
            legend_y - i * 28,
            14,
            14,
            fillColor=clr,
            strokeColor=DARK,
        )
        drawing.add(rect)
        txt = String(
            legend_x + 20,
            legend_y - i * 28 + 2,
            f"{label}: {count}",
            fontSize=10,
            fillColor=colors.HexColor("#e2e8f0"),
        )
        drawing.add(txt)

    story.append(drawing)
    story.append(Spacer(1, 0.4 * cm))
    return story


# ── Results table ─────────────────────────────────────────────────────────
def _build_results_table(styles: dict, df: pd.DataFrame) -> list:
    """Build the detailed per-image results table section.

    Args:
        styles: Dict of ParagraphStyle objects.
        df: Results DataFrame with one row per image.

    Returns:
        List of ReportLab flowable elements for the results table.
    """
    story = []
    story.append(Paragraph("Resultados por Imagen", styles["section"]))
    story.append(
        HRFlowable(
            width="100%", thickness=0.5, color=SURFACE, spaceAfter=8
        )
    )

    headers = [
        "#", "Archivo", "Estado", "Prediccion",
        "P(IA)", "P(Real)", "Preproc ms", "Infer ms",
    ]
    col_widths = [
        0.8 * cm, 5.5 * cm, 1.6 * cm, 2.0 * cm,
        1.4 * cm, 1.6 * cm, 1.8 * cm, 1.6 * cm,
    ]

    rows = [headers]
    for i, row in df.iterrows():
        filename = str(row.get("filename", ""))
        if len(filename) > 32:
            filename = filename[:30] + "..."

        status = str(row.get("status", ""))
        pred = (
            str(row.get("predicted_label", ""))
            if row.get("predicted_label")
            else "-"
        )
        prob_ai_val = row.get("prob_ai")
        p_ai = (
            f'{float(row["prob_ai"]):.2f}'
            if prob_ai_val is not None and str(prob_ai_val) != "None"
            else "-"
        )
        prob_real_val = row.get("prob_real")
        p_real = (
            f'{float(row["prob_real"]):.2f}'
            if prob_real_val is not None and str(prob_real_val) != "None"
            else "-"
        )
        pre_val = row.get("preprocess_time_ms")
        pre_ms = str(pre_val) if pre_val is not None else "-"
        inf_val = row.get("inference_time_ms")
        inf_ms = str(inf_val) if inf_val is not None else "-"

        rows.append(
            [str(int(i) + 1), filename, status, pred,
             p_ai, p_real, pre_ms, inf_ms]
        )

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)

    row_colors = []
    for idx in range(1, len(rows)):
        bg = SURFACE if idx % 2 == 1 else DARK
        row_colors.append(("BACKGROUND", (0, idx), (-1, idx), bg))

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("TEXTCOLOR", (0, 1), (-1, -1), TEXT_LIGHT),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 7.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (1, 1), (1, -1), "LEFT"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 0.5, ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 1, ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [SURFACE, DARK]),
    ]

    # Color error rows in soft red, highlight prediction labels
    for idx, row in enumerate(rows[1:], start=1):
        if row[2] == "error":
            style_cmds.append(("TEXTCOLOR", (2, idx), (2, idx), ERROR_COLOR))
        elif row[2] == "ok":
            style_cmds.append(("TEXTCOLOR", (2, idx), (2, idx), SUCCESS))
        if row[3] == "ai":
            style_cmds.append(("TEXTCOLOR", (3, idx), (3, idx), ACCENT))
        elif row[3] == "real":
            style_cmds.append(("TEXTCOLOR", (3, idx), (3, idx), SUCCESS))

    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)
    return story


# ── Public entry point ────────────────────────────────────────────────────
def build_pdf_bytes(df: pd.DataFrame) -> bytes:
    """Generate the PDF report and return the bytes ready for download.

    Args:
        df: Results DataFrame with one row per analyzed image.

    Returns:
        Bytes of the generated PDF document.
    """
    buffer = io.BytesIO()
    styles = _styles()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
    )

    story = []
    story.extend(_build_cover(styles, df))
    story.extend(_build_disclaimer(styles))
    story.extend(_build_summary(styles, df))
    story.append(PageBreak())
    story.extend(_build_results_table(styles, df))

    doc.build(
        story,
        onFirstPage=_dark_background,
        onLaterPages=_dark_background,
    )
    return buffer.getvalue()