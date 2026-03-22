"""PDF report generator for PCB defect inspection - Flux Solutions Cali.

Responsibility: build and return PDF bytes from a results DataFrame and
optionally embed processed PCB images with detected defects highlighted.
Has no Streamlit dependencies.
"""
from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from typing import Any, List, Optional

import pandas as pd
from reportlab.graphics import renderPDF  # noqa: F401
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
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
            fontSize=24,
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
        "img_caption": ParagraphStyle(
            "img_caption",
            fontName="Helvetica",
            fontSize=8,
            textColor=TEXT_MUTED,
            alignment=TA_CENTER,
            spaceAfter=4,
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
        f"Inspección de Calidad PCB - Flux Solutions Cali  |  "
        f"Reporte generado el {date_str}  |  "
        f"Página {doc.page}"
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
    story.append(Spacer(1, 3.0 * cm))

    story.append(Paragraph("Inspección de Calidad de PCB", styles["title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "Flux Solutions Cali",
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            "Reporte de Detección de Defectos — YOLOv8",
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
    aprobadas = len(df[df["Estado"] == "Aprobado"])
    rechazadas = len(df[df["Estado"] == "Rechazado"])

    stats = [
        [
            Paragraph(str(total), styles["stat_value"]),
            Paragraph(str(aprobadas), styles["stat_value"]),
            Paragraph(str(rechazadas), styles["stat_value"]),
        ],
        [
            Paragraph("Total PCBs", styles["stat_label"]),
            Paragraph("Aprobadas", styles["stat_label"]),
            Paragraph("Rechazadas", styles["stat_label"]),
        ],
    ]

    tbl = Table(stats, colWidths=[5 * cm, 5 * cm, 5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
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
        "Esta herramienta es <b>de apoyo</b> para inspección preliminar de PCB.",
        "<b>No</b> reemplaza la verificación manual por personal calificado.",
        (
            "Los resultados son <b>probabilísticos</b> y pueden contener "
            "falsos positivos/negativos."
        ),
        "No usar como única base para decisiones de rechazo en línea de producción.",
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
    story.append(Paragraph("Resumen Estadístico", styles["section"]))
    story.append(
        HRFlowable(
            width="100%", thickness=0.5, color=SURFACE, spaceAfter=8
        )
    )

    total = len(df)
    aprobadas = len(df[df["Estado"] == "Aprobado"])
    rechazadas = len(df[df["Estado"] == "Rechazado"])
    tasa = f"{(aprobadas / total * 100):.1f}%" if total > 0 else "N/A"

    rows = [
        ["Métrica", "Valor"],
        ["Total de PCBs analizadas", str(total)],
        ["PCBs aprobadas (sin defectos)", str(aprobadas)],
        ["PCBs rechazadas (con defectos)", str(rechazadas)],
        ["Tasa de aprobación", tasa],
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
    return story


# ── Results table ─────────────────────────────────────────────────────────
def _build_results_table(styles: dict, df: pd.DataFrame) -> list:
    """Build the detailed per-PCB results table section.

    Args:
        styles: Dict of ParagraphStyle objects.
        df: Results DataFrame with one row per image.

    Returns:
        List of ReportLab flowable elements for the results table.
    """
    story = []
    story.append(Paragraph("Resultados por PCB", styles["section"]))
    story.append(
        HRFlowable(
            width="100%", thickness=0.5, color=SURFACE, spaceAfter=8
        )
    )

    headers = ["#", "Archivo", "Estado", "Hallazgos", "Tiempo"]
    col_widths = [0.8 * cm, 4.5 * cm, 2.2 * cm, 7.0 * cm, 2.0 * cm]

    rows = [headers]
    for i, row in df.iterrows():
        filename = str(row.get("Nombre Archivo", ""))
        if len(filename) > 38:
            filename = filename[:36] + "..."

        estado = str(row.get("Estado", ""))
        hallazgos = str(row.get("Hallazgos", ""))
        if len(hallazgos) > 60:
            hallazgos = hallazgos[:58] + "..."
        tiempo = str(row.get("Tiempo Inferencia", "-"))

        rows.append([str(int(i) + 1), filename, estado, hallazgos, tiempo])

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)

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
        ("ALIGN", (3, 1), (3, -1), "LEFT"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 0.5, ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 1, ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [SURFACE, DARK]),
    ]

    for idx, row in enumerate(rows[1:], start=1):
        if row[2] == "Rechazado":
            style_cmds.append(("TEXTCOLOR", (2, idx), (2, idx), ERROR_COLOR))
        elif row[2] == "Aprobado":
            style_cmds.append(("TEXTCOLOR", (2, idx), (2, idx), SUCCESS))

    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)
    return story


# ── Processed images gallery ──────────────────────────────────────────────
def _build_images_gallery(
    styles: dict,
    batch_items: Optional[List[Any]],
) -> list:
    """Build the processed PCB images gallery section.

    Embeds the annotated images (with bounding boxes) from the session state
    so the technician can spatially locate each defect.

    Args:
        styles: Dict of ParagraphStyle objects.
        batch_items: List of BatchImage instances from session state,
            or None if not available.

    Returns:
        List of ReportLab flowable elements for the images section.
    """
    if not batch_items:
        return []

    done_items = [
        it for it in batch_items
        if getattr(it, "status", None) == "done"
        and getattr(it, "processed_image_base64", None)
    ]

    if not done_items:
        return []

    story = []
    story.append(PageBreak())
    story.append(Paragraph("Galería de PCBs Inspeccionadas", styles["section"]))
    story.append(
        HRFlowable(
            width="100%", thickness=0.5, color=SURFACE, spaceAfter=8
        )
    )
    story.append(
        Paragraph(
            "Las imágenes procesadas muestran las detecciones del modelo "
            "YOLOv8 con bounding boxes indicando la ubicación espacial "
            "de cada defecto detectado.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    page_width = A4[0] - 3.6 * cm  # total usable width
    img_width = (page_width - 0.5 * cm) / 2  # two images per row
    img_height = img_width * 0.75  # 4:3 ratio

    for item in done_items:
        story.append(
            Paragraph(f"<b>{item.filename}</b>", styles["body"])
        )

        row_data = []
        captions = []

        # Original image
        if getattr(item, "content", None):
            try:
                orig_buf = io.BytesIO(item.content)
                rl_orig = RLImage(orig_buf, width=img_width, height=img_height)
                row_data.append(rl_orig)
                captions.append("Imagen Original")
            except Exception:
                row_data.append(Paragraph("[ sin imagen ]", styles["img_caption"]))
                captions.append("Imagen Original")
        else:
            row_data.append(Paragraph("[ sin imagen ]", styles["img_caption"]))
            captions.append("Imagen Original")

        # Processed image (with bounding boxes)
        try:
            proc_bytes = base64.b64decode(item.processed_image_base64)
            proc_buf = io.BytesIO(proc_bytes)
            rl_proc = RLImage(proc_buf, width=img_width, height=img_height)
            row_data.append(rl_proc)
            captions.append("Imagen Procesada (Defectos)")
        except Exception:
            row_data.append(Paragraph("[ sin imagen procesada ]", styles["img_caption"]))
            captions.append("Imagen Procesada (Defectos)")

        img_table = Table(
            [row_data, [Paragraph(c, styles["img_caption"]) for c in captions]],
            colWidths=[img_width, img_width],
        )
        img_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), DARK),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(img_table)

        # Defect findings
        has_defects = getattr(item, "has_defects", None)
        defects_summary = getattr(item, "defects_summary", None) or []
        if has_defects is False:
            story.append(
                Paragraph(
                    "✓ PCB en estado óptimo. Ausencia de defectos.",
                    styles["body"],
                )
            )
        elif defects_summary:
            findings = ", ".join(
                f"{d['class']} (conf: {d['confidence']:.2f})"
                for d in defects_summary
            )
            story.append(
                Paragraph(
                    f"⚠ Defectos detectados: {findings}",
                    styles["body"],
                )
            )

        story.append(Spacer(1, 0.6 * cm))

    return story


# ── Public entry point ────────────────────────────────────────────────────
def build_pdf_bytes(
    df: pd.DataFrame,
    batch_items: Optional[List[Any]] = None,
) -> bytes:
    """Generate the PCB inspection PDF report and return the bytes.

    Args:
        df: Results DataFrame with one row per analyzed PCB.
        batch_items: Optional list of BatchImage instances from session_state.
            When provided, the PDF will include the processed images gallery
            so the technician can spatially locate each defect.

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
    story.extend(_build_images_gallery(styles, batch_items))

    doc.build(
        story,
        onFirstPage=_dark_background,
        onLaterPages=_dark_background,
    )
    return buffer.getvalue()