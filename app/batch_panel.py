"""Visual component module for batch status panel display.

This module provides UI components to render the status of batch
image processing, including thumbnails, predictions, and error states.
"""
import io
from typing import List

import streamlit as st
from PIL import Image

from batch_upload import BatchImage

STATUS_LABELS = {
    "pending": ("PENDIENTE", "badge-pending"),
    "processing": ("PROCESANDO", "badge-processing"),
    "done": ("EXITO", "badge-done"),
    "error": ("ERROR", "badge-error"),
}

BADGE_CSS = """
<style>
    .badge {
        font-size: 12px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.5px;
    }
    .badge-pending {
        background: #2a2a2a;
        color: #888;
        border: 1px solid #444;
    }
    .badge-processing {
        background: #1a3a5c;
        color: #5bc0ff;
        border: 1px solid #2d6a9f;
    }
    .badge-done {
        background: #1a3a2a;
        color: #4dff91;
        border: 1px solid #2d7a4f;
    }
    .badge-error {
        background: #3a1a1a;
        color: #ff6b6b;
        border: 1px solid #7a2d2d;
    }
    .single-filename {
        font-size: 15px;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 6px;
    }
    .single-meta {
        font-size: 12px;
        color: #94a3b8;
        margin-bottom: 4px;
    }
</style>
"""


def inject_styles() -> None:
    """Inject badge CSS styles into the Streamlit page."""
    st.markdown(BADGE_CSS, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    """Generate an HTML badge for the given status.

    Args:
        status: The status string (pending, processing, done, error).

    Returns:
        HTML string with styled badge span element.
    """
    label, css = STATUS_LABELS.get(
        status, (status.upper(), "badge-pending")
    )
    return f'<span class="badge {css}">{label}</span>'


def _render_single(item: BatchImage) -> None:
    """Render expanded view for a single image in the batch.

    Args:
        item: The BatchImage instance to display.
    """
    st.subheader("Estado del lote")

    col_img, col_info = st.columns([1, 2])

    with col_img:
        if item.content:
            try:
                img = Image.open(io.BytesIO(item.content))
                st.image(img, use_container_width=True)
            except Exception:
                st.write("[ sin preview ]")

    with col_info:
        st.markdown(
            f'<div class="single-filename">{item.filename}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(status_badge(item.status), unsafe_allow_html=True)

        if item.status == "error" and item.error_message:
            st.error(item.error_message)
        elif item.status == "done":
            if item.predicted_label:
                label = (
                    "Generada por IA"
                    if item.predicted_label == "ai"
                    else "Imagen Real"
                )
                st.markdown(
                    f'<div class="single-meta">'
                    f'Prediccion: <b>{label}</b></div>',
                    unsafe_allow_html=True,
                )
            if item.prob_ai is not None and item.prob_real is not None:
                st.markdown(
                    f'<div class="single-meta">'
                    f'P(IA): {item.prob_ai:.2f} &nbsp;|&nbsp; '
                    f'P(Real): {item.prob_real:.2f}</div>',
                    unsafe_allow_html=True,
                )
            if item.inference_time_ms is not None:
                st.markdown(
                    f'<div class="single-meta">'
                    f'Tiempo de inferencia: '
                    f'{item.inference_time_ms} ms</div>',
                    unsafe_allow_html=True,
                )
        elif item.status == "pending":
            st.markdown(
                '<div class="single-meta">Esperando analisis...</div>',
                unsafe_allow_html=True,
            )
        elif item.status == "processing":
            st.markdown(
                '<div class="single-meta">Procesando imagen...</div>',
                unsafe_allow_html=True,
            )

    st.divider()


def _render_batch(items: List[BatchImage]) -> None:
    """Render compact view for a batch of multiple images.

    Args:
        items: List of BatchImage instances to display.
    """
    st.subheader("Estado del lote")
    for item in items:
        cols = st.columns([1, 4, 2])
        with cols[0]:
            if item.content:
                try:
                    img = Image.open(io.BytesIO(item.content))
                    st.image(img, width=72)
                except Exception:
                    st.write("[ sin preview ]")
        with cols[1]:
            st.markdown(f"**{item.filename}**")
            if item.status == "error" and item.error_message:
                st.caption(item.error_message)
        with cols[2]:
            st.markdown(status_badge(item.status), unsafe_allow_html=True)
        st.divider()


def render_batch_panel(items: List[BatchImage]) -> None:
    """Render thumbnail, filename and status for each image in the batch.

    Delegates to single or batch view depending on the number of items.

    Args:
        items: List of BatchImage instances to display.
    """
    if not items:
        st.info("No hay imagenes en el lote.")
        return

    if len(items) == 1:
        _render_single(items[0])
    else:
        _render_batch(items)