"""Results table builder module for PCB defect inspection.

This module converts batch processing results into a pandas DataFrame
following the PCB inspection schema, and provides CSV export functionality.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


SCHEMA_COLUMNS = [
    "Nombre Archivo",
    "Estado",
    "Hallazgos",
    "Tiempo Inferencia",
]


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
    """Convert batch items to a DataFrame following the PCB inspection schema.

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

        Args:
            items: List of batch items to convert.

        Returns:
            DataFrame with columns: Nombre Archivo, Estado, Hallazgos,
            Tiempo Inferencia.
        """
        rows: List[Dict[str, Any]] = []

        for it in items:
            d = self._to_dict(it)

            filename = d.get("filename") or d.get("name") or "unknown"
            ui_status = d.get("status")
            has_defects = d.get("has_defects")
            defects_summary = d.get("defects_summary") or []
            inference_time_ms = d.get("inference_time_ms")
            error_message = d.get("error_message")

            # Determine approval state
            if ui_status == "error":
                estado = "Rechazado"
                hallazgos = error_message or "Error desconocido"
            elif has_defects is True:
                estado = "Rechazado"
                if defects_summary:
                    hallazgos = ", ".join(
                        f"{d['class']} ({d['confidence']:.2f})"
                        for d in defects_summary
                    )
                else:
                    hallazgos = "Defectos detectados"
            elif has_defects is False:
                estado = "Aprobado"
                hallazgos = "Sin defectos"
            else:
                estado = "Pendiente"
                hallazgos = "No procesado"

            # Format inference time
            if inference_time_ms is not None:
                try:
                    tiempo = f"{float(inference_time_ms):.1f} ms"
                except (TypeError, ValueError):
                    tiempo = str(inference_time_ms)
            else:
                tiempo = "-"

            rows.append(
                {
                    "Nombre Archivo": filename,
                    "Estado": estado,
                    "Hallazgos": hallazgos,
                    "Tiempo Inferencia": tiempo,
                }
            )

        df = pd.DataFrame(rows)

        for c in self.columns:
            if c not in df.columns:
                df[c] = None
        df = df[self.columns]

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
        return {k: getattr(it, k) for k in dir(it) if not k.startswith("_")}