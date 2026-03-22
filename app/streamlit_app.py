"""Streamlit app - AI vs Real image classifier.

Main orchestrator. Connects modules without containing business logic.

Run with:
    streamlit run app/streamlit_app.py
"""
import streamlit as st

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

st.set_page_config(page_title="AI vs Real Image Detector", layout="wide")
inject_styles()

render_header()
render_disclaimer()

client = render_sidebar()

# --- 1) Image upload ---
st.divider()
st.header("1) Carga de imagenes")

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
st.header("2) Analisis")

items = store.items()
builder = ResultsTableBuilder()

if items:
    if client is None:
        st.warning(
            "No hay conexion al servidor gRPC. "
            "Verifica que este corriendo."
        )
        render_batch_panel(items)
    else:
        if st.button("Analizar imagenes"):
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

        if results_df is not None and not results_df.empty:
            st.subheader("Resultados")
            st.dataframe(results_df, use_container_width=True)

            # --- 3) Export ---
            render_export_section(results_df, builder)
else:
    st.session_state[RESULTS_DF_KEY] = None
    st.session_state[ANALYSIS_SUMMARY_KEY] = None
    st.info("Sube imagenes para comenzar.")
