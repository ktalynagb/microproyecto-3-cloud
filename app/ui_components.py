"""Reusable UI component module for the PCB inspection app.

Responsibility: render static or configuration UI elements.
Contains no business logic or session state management.
"""
import pandas as pd
import streamlit as st

from api_client import APIClient, APIClientError
from report_pdf import build_pdf_bytes


def render_header() -> None:
    """Render the app title and general description."""
    st.title("Inspección de Calidad de PCB - Flux Solutions")
    st.write(
        "Sistema de inspección de calidad basada en visión artificial para "
        "**detección de defectos en placas de circuito impreso (PCB)** "
        "usando el modelo YOLOv8. Identifica fallas como "
        "**Dry_joint, Incorrect_installation, PCB_damage, Short_circuit, "
        "Mousebites y Opens**."
    )


def render_disclaimer() -> None:
    """Render the responsible use disclaimer warning."""
    st.warning(
        "⚠️ **Disclaimer de uso responsable**\n\n"
        "- Esta herramienta es **de apoyo** para inspección preliminar de PCB.\n"
        "- **No** reemplaza la verificación manual por personal calificado.\n"
        "- Los resultados son **probabilísticos** y pueden contener "
        "falsos positivos/negativos.\n"
        "- No usar como única base para decisiones de rechazo en línea de producción."
    )


def render_sidebar() -> APIClient | None:
    """Render the API configuration sidebar.

    Returns:
        An APIClient instance, or None if configuration is invalid.
    """
    with st.sidebar:
        st.header("Conexión API")
        host = st.text_input("Host (vacío = localhost)", value="")
        port_str = st.text_input("Puerto (vacío = 8000)", value="")
        timeout_str = st.text_input(
            "Timeout en segundos (vacío = 30)", value=""
        )

    try:
        return APIClient(
            host=host or None,
            port=int(port_str) if port_str else None,
            timeout=int(timeout_str) if timeout_str else None,
        )
    except Exception as err:
        st.error(f"Error al configurar el cliente API: {err}")
        return None


def render_summary(summary: dict) -> None:
    """Render the analysis results summary after batch processing.

    Args:
        summary: Dict with keys 'exitosas', 'fallidas', and 'total'.
    """
    exitosas = summary["exitosas"]
    fallidas = summary["fallidas"]
    total = summary["total"]

    if fallidas == 0:
        st.success(
            f"Análisis completado: {exitosas} de {total} "
            "PCBs procesadas correctamente."
        )
    elif exitosas == 0:
        st.error(
            f"No se pudo procesar ninguna imagen. "
            f"{fallidas} de {total} fallaron."
        )
    else:
        st.warning(
            f"{fallidas} de {total} imágenes no pudieron procesarse."
        )
        st.success(
            f"{exitosas} de {total} PCBs procesadas correctamente."
        )


def render_export_section(df: pd.DataFrame, builder, batch_items=None) -> None:
    """Render the export section with CSV and PDF download buttons.

    Args:
        df: Results DataFrame to export.
        builder: ResultsTableBuilder instance used for CSV export.
        batch_items: Optional list of BatchImage instances for PDF image gallery.
    """
    st.divider()
    st.header("3) Exportación")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Descargar CSV",
            data=builder.to_csv_bytes(df),
            file_name="resultados_inspeccion_pcb.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        pdf_bytes = build_pdf_bytes(df, batch_items=batch_items)
        st.download_button(
            label="Descargar PDF",
            data=pdf_bytes,
            file_name="reporte_inspeccion_pcb.pdf",
            mime="application/pdf",
            use_container_width=True,
        )