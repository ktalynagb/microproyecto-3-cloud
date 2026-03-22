"""Batch inference orchestrator module.

This module orchestrates the gRPC inference calls over a batch of images,
updating each image's status and collecting the final summary.
"""
import streamlit as st

from batch_upload import BatchStore
from batch_panel import render_batch_panel
from clientGrpc import GRPCClient
from result_table import utc_now_iso


class BatchRunner:
    """Run inference on all images in a batch, updating their state.

    Iterates over the batch, calls the gRPC client for each image,
    and populates the inference fields on each BatchImage instance.
    """

    def __init__(self, store: BatchStore, client: GRPCClient) -> None:
        """Initialize the runner with a batch store and gRPC client.

        Args:
            store: The BatchStore holding the images to process.
            client: The GRPCClient used to call the inference service.
        """
        self.store = store
        self.client = client

    def run(self) -> dict:
        """Execute inference over the batch.

        Resets each image to pending status, then processes them
        sequentially via the gRPC client. Returns a summary dict.

        Returns:
            A dict with keys 'exitosas', 'fallidas', and 'total'.
        """
        items = self.store.items()

        # Reset fields to allow re-analysis
        for item in items:
            if item.status != "error" or item.content:
                item.status = "pending"
                item.timestamp = None
                item.predicted_label = None
                item.prob_ai = None
                item.prob_real = None
                item.preprocess_time_ms = None
                item.inference_time_ms = None
                item.error_message = None

        panel_placeholder = st.empty()

        for item in items:
            # Skip images that failed to load (no content)
            if item.status == "error" and not item.content:
                continue

            item.status = "processing"
            with panel_placeholder.container():
                render_batch_panel(items)

            result = self.client.classify_image_safe(
                item.content,
                filename=item.filename,
                image_id=item.id,
            )

            item.timestamp = utc_now_iso()
            item.predicted_label = result.get("predicted_label")
            item.prob_ai = result.get("prob_ai")
            item.prob_real = result.get("prob_real")
            item.preprocess_time_ms = result.get("preprocess_time_ms")
            item.inference_time_ms = result.get("inference_time_ms")

            if result.get("status") == "error":
                item.status = "error"
                item.error_message = result.get(
                    "error_message", "Error desconocido"
                )
            else:
                item.status = "done"

            with panel_placeholder.container():
                render_batch_panel(items)

        # Final summary
        exitosas = sum(
            1 for i in self.store.items() if i.status == "done"
        )
        fallidas = sum(
            1 for i in self.store.items() if i.status == "error"
        )

        return {
            "exitosas": exitosas,
            "fallidas": fallidas,
            "total": exitosas + fallidas,
        }