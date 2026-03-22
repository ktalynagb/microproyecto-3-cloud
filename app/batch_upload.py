"""Batch upload and storage module.

This module defines the data structures and classes for managing a batch
of images in the Streamlit session state, including upload handling,
deduplication, and basic validation.
"""
import uuid
from dataclasses import dataclass
from typing import List, Literal, Optional

import streamlit as st


Status = Literal["pending", "processing", "done", "error"]


@dataclass
class BatchImage:
    """Estructura interna mínima para cada imagen del lote."""
    id: str
    filename: str
    content: bytes
    status: Status = "pending"
    error_message: Optional[str] = None
    
    # Campos E1 (se llenan luego cuando haya inferencia)
    timestamp: Optional[str] = None
    predicted_label: Optional[str] = None
    prob_ai: Optional[float] = None
    prob_real: Optional[float] = None
    preprocess_time_ms: Optional[float] = None
    inference_time_ms: Optional[float] = None


class BatchStore:
    """
    Maneja el almacenamiento del lote en st.session_state para que persista
    entre reruns de Streamlit.
    """
    KEY = "batch_images"

    def __init__(self) -> None:
        """Initialize the store, creating the session state key if absent."""
        if self.KEY not in st.session_state:
            st.session_state[self.KEY] = []

    def items(self) -> List[BatchImage]:
        """Return the current list of BatchImage objects in the batch.

        Returns:
            List of BatchImage instances stored in session state.
        """
        return st.session_state[self.KEY]

    def clear(self) -> None:
        """Remove all images from the batch."""
        st.session_state[self.KEY] = []

    def remove(self, image_id: str) -> None:
        """Remove a single image from the batch by its ID.

        Args:
            image_id: The unique identifier of the image to remove.
        """
        st.session_state[self.KEY] = [
            x for x in self.items() if x.id != image_id
        ]

    def add_uploaded_files(self, uploaded_files) -> None:
        """Add uploaded files to the batch, avoiding simple duplicates.

        Deduplication is based on (filename, size_bytes) pairs.
        Streamlit already filters by extension, but this method also
        validates that the file content is readable and non-empty.

        Args:
            uploaded_files: List of Streamlit UploadedFile objects.
        """
        if not uploaded_files:
            return

        existing_keys = {(x.filename, len(x.content)) for x in self.items()}

        for f in uploaded_files:
            try:
                content = f.getvalue()
                if not content:
                    # archivo vacío o lectura fallida
                    self.items().append(
                        BatchImage(
                            id=str(uuid.uuid4()),
                            filename=f.name,
                            content=b"",
                            status="error",
                            error_message="Archivo vacío o no se pudo leer.",
                        )
                    )
                    continue

                key = (f.name, len(content))
                if key in existing_keys:
                    continue

                self.items().append(
                    BatchImage(
                        id=str(uuid.uuid4()),
                        filename=f.name,
                        content=content,
                        status="pending",
                    )
                )
                existing_keys.add(key)

            except Exception as e:
                self.items().append(
                    BatchImage(
                        id=str(uuid.uuid4()),
                        filename=getattr(f, "name", "unknown"),
                        content=b"",
                        status="error",
                        error_message=str(e),
                    )
                )


class BatchUploader:
    """UI component for image file upload and batch management.

    Renders the Streamlit file uploader widget and provides controls
    to clear the batch or remove individual images.
    """

    # Session state key that holds the counter to reset the file uploader.
    _UPLOADER_COUNTER_KEY = "uploader_counter"

    def __init__(self, store: BatchStore) -> None:
        """Initialize the uploader with a batch store.

        Args:
            store: The BatchStore to add uploaded images to.
        """
        self.store = store
        if self._UPLOADER_COUNTER_KEY not in st.session_state:
            st.session_state[self._UPLOADER_COUNTER_KEY] = 0

    def render(self) -> None:
        """Render the file uploader widget and batch management controls."""
        counter = st.session_state[self._UPLOADER_COUNTER_KEY]
        uploaded_files = st.file_uploader(
            "Sube una o varias imágenes (JPG/JPEG/PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"file_uploader_{counter}",
        )

        # Validación básica de "existencia/carga"
        if uploaded_files:
            self.store.add_uploaded_files(uploaded_files)

        # Controles y resumen
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"**Imágenes en lote:** {len(self.store.items())}")
        with col2:
            if st.button("Limpiar lote", use_container_width=True):
                self.store.clear()
                # Increment the counter so the file_uploader widget resets
                st.session_state[self._UPLOADER_COUNTER_KEY] += 1
                st.rerun()

        # Lista simple del lote (nombre + estado inicial)
        if not self.store.items():
            st.caption("Aún no has cargado imágenes.")
            return

        for it in self.store.items():
            row = st.columns([4, 2, 1])
            with row[0]:
                st.write(it.filename)
                if it.status == "error" and it.error_message:
                    st.caption(f"❌ {it.error_message}")
            with row[1]:
                st.write(f"Estado: **{it.status}**")
            with row[2]:
                if st.button("Quitar", key=f"rm_{it.id}"):
                    self.store.remove(it.id)
                    st.rerun()