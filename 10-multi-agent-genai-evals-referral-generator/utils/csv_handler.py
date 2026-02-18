"""CSV and Excel read/write utilities."""

from __future__ import annotations

import io
from typing import List, Optional

import pandas as pd


def load_csv(file_or_path) -> pd.DataFrame:
    """Load a CSV file from a path string or file-like object.

    Args:
        file_or_path: Either a file path string or an uploaded file object
                      (e.g., from Streamlit's file_uploader).

    Returns:
        pandas DataFrame with the trace log data.
    """
    df = pd.read_csv(file_or_path)
    return df


def ensure_eval_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ensure all evaluation columns exist in the DataFrame.

    Adds missing columns with empty string values.
    """
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df


def save_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download.

    Returns:
        UTF-8 encoded CSV bytes.
    """
    return df.to_csv(index=False).encode("utf-8")


def save_excel(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to Excel bytes with formatting.

    Applies conditional formatting for score columns:
    - Green for PASS/COMPLETE/VALID/ALL_ACTIVE/NONE_MISSING and scores 4-5
    - Yellow for NEEDS_REVISION/PARTIAL/HOMEPAGE_ONLY/SOME_CHANGES/MINOR_GAPS and score 3
    - Red for FAIL/INCOMPLETE/INACCURATE/BROKEN/HAS_CLOSURES/MAJOR_GAPS and scores 1-2

    Returns:
        Excel file bytes.
    """
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Evaluations")

        workbook = writer.book
        worksheet = writer.sheets["Evaluations"]

        # Define formats
        green_fmt = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        yellow_fmt = workbook.add_format({"bg_color": "#FFEB9C", "font_color": "#9C5700"})
        red_fmt = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#4472C4",
            "font_color": "#FFFFFF",
            "border": 1,
        })

        # Format headers
        for col_num, value in enumerate(df.columns):
            worksheet.write(0, col_num, value, header_fmt)

        # Auto-fit column widths
        for col_num, col_name in enumerate(df.columns):
            max_len = max(
                df[col_name].astype(str).str.len().max() if len(df) > 0 else 0,
                len(col_name),
            )
            # Cap at 50 chars wide, min 12
            worksheet.set_column(col_num, col_num, min(max(max_len + 2, 12), 50))

        # Score columns that need conditional formatting
        score_columns = [
            "referral_service_area_eligibility",
            "referral_location_proximity",
            "referral_contact_info",
            "referral_URL_check",
            "referral_description_review",
            "referral_missing_resources",
            "referral_relevance_review",
            "referral_service_status",
            "referral_overall_review",
            "actionplan_overallreview",
        ]

        for col_name in score_columns:
            if col_name not in df.columns:
                continue
            col_idx = df.columns.get_loc(col_name)
            col_letter = _col_letter(col_idx)
            data_range = f"{col_letter}2:{col_letter}{len(df) + 1}"

            # Green values
            for val in ["PASS", "COMPLETE", "VALID", "ALL_ACTIVE", "NONE_MISSING", "5", "4"]:
                worksheet.conditional_format(data_range, {
                    "type": "text",
                    "criteria": "containing",
                    "value": val,
                    "format": green_fmt,
                })

            # Yellow values
            for val in ["NEEDS_REVISION", "PARTIAL", "HOMEPAGE_ONLY", "SOME_CHANGES", "MINOR_GAPS", "3", "N/A"]:
                worksheet.conditional_format(data_range, {
                    "type": "text",
                    "criteria": "containing",
                    "value": val,
                    "format": yellow_fmt,
                })

            # Red values
            for val in ["FAIL", "INCOMPLETE", "INACCURATE", "BROKEN", "OUTDATED", "MISSING", "HAS_CLOSURES", "MAJOR_GAPS", "1", "2", "ERROR"]:
                worksheet.conditional_format(data_range, {
                    "type": "text",
                    "criteria": "containing",
                    "value": val,
                    "format": red_fmt,
                })

        # Wrap text for reasoning columns
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
        for col_name in df.columns:
            if col_name.endswith("_reasoning"):
                col_idx = df.columns.get_loc(col_name)
                worksheet.set_column(col_idx, col_idx, 50, wrap_fmt)

    buffer.seek(0)
    return buffer.getvalue()


def _col_letter(col_idx: int) -> str:
    """Convert 0-based column index to Excel column letter."""
    result = ""
    while col_idx >= 0:
        result = chr(col_idx % 26 + ord("A")) + result
        col_idx = col_idx // 26 - 1
    return result
