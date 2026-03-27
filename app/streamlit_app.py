"""Streamlit app - Inspección de Calidad de PCB - Flux Solutions Cali.

Main orchestrator. Connects modules without containing business logic.

Run with:
    streamlit run app/streamlit_app.py
    o:
    make gui
"""
import base64
import io
import time
from typing import List

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from api_client import AzureMLClient, APIClientError
from batch_panel import inject_styles, render_batch_panel
from batch_runner import BatchRunner
from batch_upload import BatchStore, BatchUploader
from result_table import ResultsTableBuilder
from ui_components import (
    render_disclaimer,
    render_export_section,
    render_header,
    render_sidebar,
    render_summary,
)


RESULTS_DF_KEY = "results_df"
ANALYSIS_SUMMARY_KEY = "analysis_summary"

# Colores RGB para cada clase de defecto
_DEFECT_COLORS = {
    "short_circuit": (255, 50, 50),
    "dry_joint": (50, 100, 255),
    "incorrect_installation": (255, 220, 0),
    "pcb_damage": (255, 50, 255),
}
_DEFAULT_COLOR = (0, 255, 120)


def _color_for_class(class_name: str) -> tuple:
    """Devuelve el color RGB para la clase de defecto dada."""
    return _DEFECT_COLORS.get(class_name.lower(), _DEFAULT_COLOR)


def _draw_bboxes(image_bytes: bytes, detections: list) -> bytes:
    """Dibuja bounding boxes sobre la imagen y devuelve los bytes resultantes.

    Args:
        image_bytes: Bytes crudos de la imagen original.
        detections: Lista de dicts con 'class', 'confidence' y 'bbox'
                    ([x1, y1, x2, y2] normalizados 0–1).

    Returns:
        Bytes de la imagen anotada en formato PNG.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    try:
        # Try common font locations; fall back to PIL default
        import os as _os
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
            "/Library/Fonts/Arial Bold.ttf",                          # macOS
            "C:/Windows/Fonts/arialbd.ttf",                           # Windows
        ]
        font = None
        for candidate in font_candidates:
            if _os.path.exists(candidate):
                font = ImageFont.truetype(candidate, 14)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)
        color = _color_for_class(det.get("class", ""))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{det.get('class', '')} {det.get('confidence', 0):.1f}%"
        draw.text((x1 + 2, max(0, y1 - 16)), label, fill=color, font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _run_azure_batch(client: AzureMLClient, store: BatchStore) -> dict:
    """Ejecuta el flujo Azure ML Batch: submit → polling → results.

    Args:
        client: AzureMLClient configurado con la URL del backend.
        store: BatchStore con las imágenes a procesar.

    Returns:
        Dict con 'images' (lista de resultados) y 'summary'.
    """
    items = store.items()
    if not items:
        return {}

    # Preparar listas de bytes y nombres
    image_bytes_list: List[bytes] = [it.content for it in items if it.content]
    filenames: List[str] = [it.filename for it in items if it.content]

    # ── 1. Enviar lote ────────────────────────────────────────────────────
    status_ph = st.empty()
    status_ph.info("📤 Enviando imágenes al backend Azure ML...")

    try:
        job_id = client.submit_inference(image_bytes_list, filenames)
    except APIClientError as exc:
        st.error(f"❌ Error al enviar el lote: {exc}")
        return {}

    st.success(f"✅ Lote enviado. Job ID: `{job_id}`")

    # ── 2. Polling ────────────────────────────────────────────────────────
    poll_ph = st.empty()
    progress_bar = st.progress(0)
    poll_interval = 10  # segundos
    max_polls = 60  # ~10 minutos máximo

    for attempt in range(max_polls):
        try:
            status_info = client.poll_results(job_id)
        except APIClientError as exc:
            st.error(f"❌ Error consultando estado: {exc}")
            return {}

        current_status = status_info.get("status", "unknown")
        poll_ph.info(
            f"⏳ Estado del job: **{current_status.upper()}** "
            f"(verificación {attempt + 1}/{max_polls})"
        )
        progress_bar.progress(min((attempt + 1) / max_polls, 1.0))

        if current_status == "completed":
            poll_ph.success("✅ ¡Job completado!")
            break
        if current_status == "failed":
            poll_ph.error("❌ El job falló en Azure ML. Revisa los logs.")
            return {}

        time.sleep(poll_interval)
    else:
        poll_ph.warning("⚠️ Timeout esperando resultados. Intenta más tarde.")
        return {}

    # ── 3. Obtener resultados ─────────────────────────────────────────────
    status_ph.info("📥 Descargando resultados...")
    try:
        results = client.get_download_links(job_id)
    except APIClientError as exc:
        st.error(f"❌ Error obteniendo resultados: {exc}")
        return {}

    status_ph.empty()
    return results


st.set_page_config(
    page_title="Inspección de Calidad PCB - Flux Solutions",
    layout="wide",
)
inject_styles()

render_header()
render_disclaimer()

client = render_sidebar()

# --- 1) Image upload ---
st.divider()
st.header("1) Carga de imágenes PCB")

if "store" not in st.session_state:
    st.session_state.store = BatchStore()

store = st.session_state.store
if RESULTS_DF_KEY not in st.session_state:
    st.session_state[RESULTS_DF_KEY] = None
if ANALYSIS_SUMMARY_KEY not in st.session_state:
    st.session_state[ANALYSIS_SUMMARY_KEY] = None

uploader = BatchUploader(store)
uploader.render()

# --- 2) Analysis ---
st.divider()
st.header("2) Inspección y Análisis")

items = store.items()
builder = ResultsTableBuilder()

if items:
    if client is None:
        st.warning(
            "No hay conexión al servidor de inferencia. "
            "Verifica que el servidor FastAPI esté corriendo "
            "(make api-server)."
        )
        render_batch_panel(items)
    elif isinstance(client, AzureMLClient):
        # ── Modo Azure ML Batch ───────────────────────────────────────────
        if st.button("🚀 Analizar PCBs con Azure ML Batch"):
            azure_results = _run_azure_batch(client, store)
            if azure_results:
                st.session_state["azure_results"] = azure_results
                st.session_state[ANALYSIS_SUMMARY_KEY] = {
                    "exitosas": azure_results.get("summary", {}).get("total_images", 0),
                    "fallidas": 0,
                    "total": azure_results.get("summary", {}).get("total_images", 0),
                }
                st.rerun()

        # Mostrar resultados Azure si están disponibles
        azure_results = st.session_state.get("azure_results")
        if azure_results:
            summary = azure_results.get("summary", {})
            st.success(
                f"✅ Lote procesado: **{summary.get('total_images', 0)}** imágenes · "
                f"**{summary.get('defective_images', 0)}** con defectos · "
                f"**{summary.get('total_defects', 0)}** defectos totales"
            )

            images_data = azure_results.get("images", [])
            if images_data:
                st.subheader("Galería de Resultados – Bounding Boxes")
                # Build a name→content map from the batch store
                content_map = {it.filename: it.content for it in store.items() if it.content}

                for img_res in images_data:
                    filename = img_res.get("filename", "")
                    detections = img_res.get("detections", [])
                    has_defects = img_res.get("has_defects", False)

                    st.markdown(f"**{filename}**")
                    col_orig, col_annot = st.columns(2)

                    original_bytes = content_map.get(filename)
                    with col_orig:
                        st.caption("Imagen Original")
                        if original_bytes:
                            try:
                                st.image(Image.open(io.BytesIO(original_bytes)), use_container_width=True)
                            except Exception:
                                st.write("[ sin preview ]")

                    with col_annot:
                        st.caption("Predicciones (Bounding Boxes)")
                        if original_bytes and detections:
                            try:
                                annotated = _draw_bboxes(original_bytes, detections)
                                st.image(Image.open(io.BytesIO(annotated)), use_container_width=True)
                            except Exception as exc:
                                st.write(f"[ error al dibujar: {exc} ]")
                        elif original_bytes:
                            st.image(Image.open(io.BytesIO(original_bytes)), use_container_width=True)
                            st.caption("Sin detecciones")

                    if has_defects:
                        defect_lines = [
                            f"• **{d['class']}** – confianza: {d['confidence']:.1f}%"
                            for d in detections
                        ]
                        st.error("⚠️ Defectos detectados:\n" + "\n".join(defect_lines))
                    else:
                        st.success("✅ PCB en estado óptimo. Ausencia de defectos.")
                    st.divider()

        analysis_summary = st.session_state.get(ANALYSIS_SUMMARY_KEY)
        if analysis_summary:
            render_summary(analysis_summary)
    else:
        # ── Modo Local (inferencia sincrónica) ────────────────────────────
        if st.button("Analizar PCBs"):
            runner = BatchRunner(store=store, client=client)
            summary = runner.run()
            st.session_state[ANALYSIS_SUMMARY_KEY] = summary
            st.session_state[RESULTS_DF_KEY] = (
                builder.from_batch_items(store.items())
            )
            st.rerun()

        render_batch_panel(store.items())

        analysis_summary = st.session_state.get(ANALYSIS_SUMMARY_KEY)
        results_df = st.session_state.get(RESULTS_DF_KEY)

        if analysis_summary is not None:
            render_summary(analysis_summary)

        # Gallery: show original vs processed image side by side
        done_items = [i for i in store.items() if i.status == "done"]
        if done_items:
            st.divider()
            st.subheader("Galería de Resultados - Comparativa")
            for item in done_items:
                st.markdown(f"**{item.filename}**")
                col_orig, col_proc = st.columns(2)

                with col_orig:
                    st.caption("Imagen Original")
                    if item.content:
                        try:
                            orig_img = Image.open(io.BytesIO(item.content))
                            st.image(orig_img, use_container_width=True)
                        except Exception:
                            st.write("[ sin preview ]")

                with col_proc:
                    st.caption("Imagen Procesada (Defectos detectados)")
                    if item.processed_image_base64:
                        try:
                            proc_bytes = base64.b64decode(
                                item.processed_image_base64
                            )
                            proc_img = Image.open(io.BytesIO(proc_bytes))
                            st.image(proc_img, use_container_width=True)
                        except Exception:
                            st.write("[ sin imagen procesada ]")
                    else:
                        st.write("[ sin imagen procesada ]")

                if item.has_defects is False:
                    st.success(
                        "✅ PCB en estado óptimo. Ausencia de defectos."
                    )
                elif item.has_defects is True and item.defects_summary:
                    defects_list = ", ".join(
                        f"{d['class']} ({d['confidence']:.2f})"
                        for d in item.defects_summary
                    )
                    st.error(
                        f"⚠️ Defectos detectados: {defects_list}"
                    )
                st.divider()

        if results_df is not None and not results_df.empty:
            st.subheader("Tabla de Resultados")
            st.dataframe(results_df, use_container_width=True)

            # --- 3) Export ---
            render_export_section(results_df, builder, batch_items=store.items())
else:
    st.session_state[RESULTS_DF_KEY] = None
    st.session_state[ANALYSIS_SUMMARY_KEY] = None
    if "azure_results" in st.session_state:
        del st.session_state["azure_results"]
    st.info("Sube imágenes PCB para comenzar la inspección.")

