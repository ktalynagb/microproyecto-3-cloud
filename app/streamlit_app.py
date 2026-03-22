"""Streamlit app - Inspección de Calidad de PCB - Flux Solutions Cali.

Main orchestrator. Connects modules without containing business logic.

Run with:
    streamlit run app/streamlit_app.py
    o:
    make gui
"""
import base64
import io

import streamlit as st
from PIL import Image

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
    else:
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
    st.info("Sube imágenes PCB para comenzar la inspección.")

