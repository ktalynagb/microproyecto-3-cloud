"""Results table builder module.

This module converts batch processing results into a pandas DataFrame
following the E1 schema, and provides CSV export functionality.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


LABEL_ALIASES = {
    "ai": "ai",
    "ia": "ai",
    "artificial": "ai",
    "fake": "ai",
    "generated": "ai",
    "real": "real",
    "hum": "real",
    "human": "real",
    "humana": "real",
    "humano": "real",
}


SCHEMA_COLUMNS = [
    "timestamp",
    "filename",
    "status",
    "predicted_label",
    "prob_ai",
    "prob_real",
    "preprocess_time_ms",
    "inference_time_ms",
    "error_message",
]




def normalize_prediction_label(label: Any) -> Any:
    """Normalize model/UI aliases to canonical labels: ai | real.

    Args:
        label: Raw prediction label from the model or UI.

    Returns:
        Canonical label string ('ai' or 'real'), or None if unresolvable.
    """
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None

    normalized = str(label).strip().casefold()
    if not normalized:
        return None

    return LABEL_ALIASES.get(normalized, normalized)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string without microseconds.

    Returns:
        ISO 8601 UTC timestamp string, e.g. '2026-03-02T05:12:10Z'.
    """
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class ResultsTableBuilder:
    """Convert batch items to a DataFrame following the E1 schema.

    Supports BatchImage dataclass instances, plain dicts, and generic
    objects with public attributes.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        """Initialize the builder with an optional column order override.

        Args:
            columns: Ordered list of column names. Defaults to
                SCHEMA_COLUMNS if not provided.
        """
        self.columns = columns or SCHEMA_COLUMNS

    def from_batch_items(self, items: List[Any]) -> pd.DataFrame:
        """Convert a list of batch items to a DataFrame.

        Each item may be a BatchImage dataclass, a dict, or an object
        with public attributes. Normalizes labels and E1 status values.

        Args:
            items: List of batch items to convert.

        Returns:
            DataFrame with columns defined by self.columns.
        """
        rows: List[Dict[str, Any]] = []

        for it in items:
            # Support dataclass (BatchImage) objects or session_state dicts
            d = self._to_dict(it)

            filename = d.get("filename") or d.get("name") or "unknown"
            ui_status = d.get("status")  # pending/processing/done/error
            timestamp = d.get("timestamp") or utc_now_iso()

            # Prediction fields (may be None before inference runs)
            predicted_label = normalize_prediction_label(
                d.get("predicted_label")
            )
            prob_ai = d.get("prob_ai")
            prob_real = d.get("prob_real")
            preprocess_time_ms = d.get("preprocess_time_ms")
            inference_time_ms = d.get("inference_time_ms")
            error_message = d.get("error_message")

            # Normalize to E1 schema: status is ok or error
            if ui_status == "error":
                status = "error"
                if not error_message:
                    error_message = "Unknown error"
            elif (
                predicted_label is not None
                or prob_ai is not None
                or prob_real is not None
            ):
                # Prediction present: classify as ok
                status = "ok"
            else:
                # Not yet processed: mark as error with message
                status = "error"
                error_message = error_message or "Not processed yet"

            rows.append(
                {
                    "timestamp": timestamp,
                    "filename": filename,
                    "status": status,
                    "predicted_label": predicted_label,
                    "prob_ai": prob_ai,
                    "prob_real": prob_real,
                    "preprocess_time_ms": preprocess_time_ms,
                    "inference_time_ms": inference_time_ms,
                    "error_message": error_message,
                }
            )

        df = pd.DataFrame(rows)

        # Ensure all E1 columns exist and are in the correct order
        for c in self.columns:
            if c not in df.columns:
                df[c] = None
        df = df[self.columns]

        # Cast numeric fields that may have arrived as strings
        num_cols = [
            "prob_ai", "prob_real",
            "preprocess_time_ms", "inference_time_ms",
        ]
        for num_col in num_cols:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

        return df

    def to_csv_bytes(self, df: pd.DataFrame) -> bytes:
        """Serialize the DataFrame to CSV-encoded bytes.

        Args:
            df: DataFrame to serialize.

        Returns:
            UTF-8 encoded CSV bytes without the index column.
        """
        return df.to_csv(index=False).encode("utf-8")

    def _to_dict(self, it: Any) -> Dict[str, Any]:
        """Convert a batch item to a plain dict.

        Supports dataclass instances, plain dicts, and generic objects
        with public attributes.

        Args:
            it: The item to convert.

        Returns:
            Dict representation of the item.
        """
        if is_dataclass(it):
            return asdict(it)
        if isinstance(it, dict):
            return dict(it)
        # Fallback: use public attributes
        return {k: getattr(it, k) for k in dir(it) if not k.startswith("_")}