"""Frontend Streamlit para PCB Defect Detection.

Funcionalidades:
  - Subir una o varias imágenes de PCB
  - Enviar al pipeline de Azure ML via API REST
  - Barra de progreso durante el polling
  - Visualizar resultados: número de defectos, clases, confianzas
  - Descargar imágenes anotadas
  - Manejo de errores con mensajes claros
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from PIL import Image

from api_client import APIError, PCBApiClient

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
PCB_API_KEY = os.environ.get("PCB_API_KEY", "changeme-secret-key")

st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔬",
    layout="wide",
)

# Paleta de colores por clase de defecto
CLASS_COLORS: dict[str, str] = {
    "Missing_hole": "#FF4B4B",
    "Incorrect_installation": "#FFA500",
    "Copper_exposed": "#FFD700",
    "Short_circuit": "#4169E1",
    "Open_circuit": "#32CD32",
    "Spur": "#DA70D6",
}

# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------

if "client" not in st.session_state:
    st.session_state.client = PCBApiClient(
        base_url=API_BASE_URL, api_key=PCB_API_KEY
    )
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "results" not in st.session_state:
    st.session_state.results = None
if "api_connected" not in st.session_state:
    st.session_state.api_connected = False

client: PCBApiClient = st.session_state.client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_api_health() -> bool:
    try:
        client.health()
        return True
    except Exception:
        return False


def render_detection_badge(class_name: str, confidence: float) -> str:
    color = CLASS_COLORS.get(class_name, "#888")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em;margin:2px;display:inline-block;">'
        f"{class_name} {confidence:.1%}</span>"
    )


def render_result_card(result: dict[str, Any], idx: int) -> None:
    filename = result.get("filename", f"imagen_{idx}")
    has_defects = result.get("has_defects", False)
    detections = result.get("detections", [])
    inf_time = result.get("inference_time_ms", 0)
    error = result.get("error")

    with st.container():
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"**📄 {filename}**")
            if error:
                st.error(f"❌ {error}")
                return

            if has_defects:
                st.error(f"⚠️ {len(detections)} defecto(s) detectado(s)")
            else:
                st.success(result.get("no_defect_notification", "✅ Sin defectos"))

            st.caption(f"⏱️ Inferencia: {inf_time:.0f}ms")

        with col2:
            if detections:
                st.markdown("**Defectos encontrados:**")
                badges_html = "".join(
                    render_detection_badge(d["class_name"], d["confidence"])
                    for d in detections
                )
                st.markdown(badges_html, unsafe_allow_html=True)

                # Tabla de detecciones
                with st.expander("Ver detalles", expanded=False):
                    for i, det in enumerate(detections, 1):
                        st.markdown(
                            f"**{i}.** `{det['class_name']}` — "
                            f"Confianza: **{det['confidence']:.1%}** — "
                            f"BBox: {[f'{v:.3f}' for v in det.get('bbox', [])]}"
                        )

        st.divider()


def poll_with_progress(job_id: str) -> list[dict[str, Any]] | None:
    """Hace polling mostrando una progress bar de Streamlit."""
    progress_bar = st.progress(0.0, text="Iniciando inferencia…")
    status_placeholder = st.empty()

    try:
        results = client.wait_for_job(
            job_id,
            poll_interval=5.0,
            timeout=600.0,
            on_progress=lambda p, msg: (
                progress_bar.progress(min(p, 1.0), text=msg),
                status_placeholder.info(f"⏳ {msg}"),
            ),
        )
        progress_bar.progress(1.0, text="✅ Completado")
        status_placeholder.empty()
        return results
    except APIError as exc:
        progress_bar.empty()
        status_placeholder.empty()
        st.error(f"❌ Error en el job: {exc.detail}")
        return None
    except TimeoutError:
        progress_bar.empty()
        status_placeholder.empty()
        st.error("❌ Timeout: el job tardó demasiado.")
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuración")

    api_url = st.text_input("API URL", value=API_BASE_URL)
    api_key = st.text_input("API Key", value=PCB_API_KEY, type="password")

    if st.button("Conectar"):
        client = PCBApiClient(base_url=api_url, api_key=api_key)
        st.session_state.client = client
        if check_api_health():
            st.session_state.api_connected = True
            st.success("✅ Conectado")
        else:
            st.session_state.api_connected = False
            st.error("❌ No se pudo conectar a la API")

    # Estado de conexión
    if st.session_state.api_connected:
        st.markdown("🟢 **API:** Conectada")
    else:
        connected = check_api_health()
        st.session_state.api_connected = connected
        if connected:
            st.markdown("🟢 **API:** Conectada")
        else:
            st.markdown("🔴 **API:** Desconectada")

    st.divider()
    st.markdown("### ℹ️ Información")
    st.markdown(
        "**Clases detectables:**\n"
        + "\n".join(
            f"- <span style='color:{c}'>{cls}</span>"
            for cls, c in CLASS_COLORS.items()
        ),
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Página principal
# ---------------------------------------------------------------------------

st.title("🔬 PCB Defect Detection")
st.markdown(
    "Detecta defectos en placas de circuito impreso usando "
    "**YOLOv8** ejecutado en **Azure ML Batch**."
)

# Avisar si la API no está disponible
if not st.session_state.api_connected:
    st.warning(
        "⚠️ La API no está disponible. Configura la URL y API Key en el sidebar "
        "y pulsa **Conectar**."
    )

tab_upload, tab_results, tab_history = st.tabs(
    ["📤 Subir Imágenes", "📊 Resultados", "📜 Historial"]
)

# ---------------------------------------------------------------------------
# Tab: Subir imágenes
# ---------------------------------------------------------------------------

with tab_upload:
    st.markdown("### Selecciona las imágenes a analizar")

    uploaded_files = st.file_uploader(
        "Imágenes PCB",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        help="Sube una o varias imágenes de PCB para detectar defectos.",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} imagen(es) seleccionada(s):**")
        cols = st.columns(min(len(uploaded_files), 4))
        for i, f in enumerate(uploaded_files):
            with cols[i % 4]:
                img = Image.open(io.BytesIO(f.read()))
                f.seek(0)
                st.image(img, caption=f.name, use_column_width=True)

        st.markdown("---")

        # Advertencia si hay muchas imágenes → sugerir batch
        if len(uploaded_files) > 10:
            st.info(
                f"📦 {len(uploaded_files)} imágenes: se usará el "
                "**Batch Endpoint** (2-3 minutos)."
            )
        else:
            st.info(
                f"📤 {len(uploaded_files)} imagen(es): procesando via "
                "**Batch Endpoint**."
            )

        col_btn, col_clear = st.columns([3, 1])
        with col_btn:
            run_btn = st.button(
                "🚀 Analizar",
                type="primary",
                disabled=not st.session_state.api_connected,
                use_container_width=True,
            )
        with col_clear:
            clear_btn = st.button("🗑️ Limpiar", use_container_width=True)

        if clear_btn:
            st.session_state.results = None
            st.session_state.job_id = None
            st.rerun()

        if run_btn:
            # Guardar archivos temporalmente para el cliente
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_paths: list[str] = []
                for f in uploaded_files:
                    tmp_path = os.path.join(tmpdir, f.name)
                    with open(tmp_path, "wb") as out:
                        out.write(f.read())
                    f.seek(0)
                    tmp_paths.append(tmp_path)

                with st.spinner("Enviando imágenes…"):
                    try:
                        job_id = client.submit_batch(tmp_paths)
                        st.session_state.job_id = job_id
                        st.success(f"✅ Job enviado: `{job_id}`")
                    except APIError as exc:
                        st.error(f"❌ Error enviando: {exc.detail}")
                        job_id = None
                    except Exception as exc:
                        st.error(f"❌ Error inesperado: {exc}")
                        job_id = None

                if job_id:
                    st.markdown("### ⏳ Procesando…")
                    results = poll_with_progress(job_id)
                    if results is not None:
                        st.session_state.results = results
                        st.success(
                            f"🎉 Análisis completado: {len(results)} imagen(es)"
                        )
                        st.balloons()
                        # Cambiar a la pestaña de resultados
                        st.info("👉 Ve a la pestaña **Resultados** para ver el análisis.")

# ---------------------------------------------------------------------------
# Tab: Resultados
# ---------------------------------------------------------------------------

with tab_results:
    if not st.session_state.results:
        st.info("No hay resultados aún. Sube imágenes en la pestaña **Subir Imágenes**.")
    else:
        results = st.session_state.results
        total = len(results)
        with_defects = sum(1 for r in results if r.get("has_defects"))
        without_defects = total - with_defects

        # Métricas resumen
        st.markdown("### 📊 Resumen")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total imágenes", total)
        m2.metric("Con defectos", with_defects, delta=None)
        m3.metric("Sin defectos", without_defects)
        m4.metric(
            "Tasa de defectos",
            f"{with_defects / total:.1%}" if total > 0 else "N/A",
        )

        # Distribución de clases
        class_counts: dict[str, int] = {}
        for r in results:
            for det in r.get("detections", []):
                cls = det.get("class_name", "Unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1

        if class_counts:
            st.markdown("### 🏷️ Distribución de Defectos")
            import pandas as pd
            df = pd.DataFrame(
                [(k, v) for k, v in sorted(class_counts.items(), key=lambda x: -x[1])],
                columns=["Clase", "Conteo"],
            )
            st.bar_chart(df.set_index("Clase"))

        st.markdown("### 🖼️ Resultados por Imagen")

        # Filtros
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            filter_defects = st.selectbox(
                "Filtrar por:",
                ["Todas", "Con defectos", "Sin defectos"],
            )
        with col_filter2:
            sort_by = st.selectbox("Ordenar por:", ["Nombre", "Defectos (mayor a menor)"])

        filtered = results
        if filter_defects == "Con defectos":
            filtered = [r for r in results if r.get("has_defects")]
        elif filter_defects == "Sin defectos":
            filtered = [r for r in results if not r.get("has_defects")]

        if sort_by == "Defectos (mayor a menor)":
            filtered = sorted(
                filtered, key=lambda r: r.get("detections_count", 0), reverse=True
            )

        for i, result in enumerate(filtered, 1):
            render_result_card(result, i)

        # Descarga de resultados en JSON
        st.markdown("### 💾 Descargar Resultados")
        import json
        results_json = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Descargar JSON",
            data=results_json.encode("utf-8"),
            file_name=f"pcb_results_{st.session_state.job_id or 'export'}.json",
            mime="application/json",
        )

# ---------------------------------------------------------------------------
# Tab: Historial
# ---------------------------------------------------------------------------

with tab_history:
    st.markdown("### 📜 Historial de Jobs")

    if not st.session_state.api_connected:
        st.warning("API no conectada.")
    else:
        if st.button("🔄 Actualizar"):
            pass  # Forzar re-render

        try:
            jobs = client.list_jobs()
            if not jobs:
                st.info("No hay jobs registrados.")
            else:
                import pandas as pd
                df = pd.DataFrame(jobs)
                df["progress"] = df["progress"].apply(lambda p: f"{float(p):.0%}")
                st.dataframe(df, use_container_width=True)
        except Exception as exc:
            st.error(f"Error cargando historial: {exc}")
