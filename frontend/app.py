import base64
import hashlib
import io
import re
from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import streamlit as st
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
)



DEVELOPER_IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "arpit_pandey.png"


def get_developer_image_uri():
    """Return the developer portrait as a browser-safe data URI."""
    if not DEVELOPER_IMAGE_PATH.exists():
        return ""
    try:
        image = Image.open(DEVELOPER_IMAGE_PATH).convert("RGB")
        return pil_to_data_uri(image, "PNG")
    except Exception:
        return ""


def pil_to_data_uri(image, image_format="PNG"):
    """Convert a PIL image to a browser-safe data URI for styled HTML display."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=image_format, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{encoded}"

# ============================================================
# BACKEND
# ============================================================

BACKEND_URL = "http://localhost:8000/predict"
SCREENINGS_URL = "http://localhost:8000/screenings"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DR Screening",
    page_icon="👁️",
    layout="wide"
)

# ============================================================
# PERSISTENT SCREENING STATE
# ============================================================
# Streamlit reruns this script when the user changes navigation.
# Keep the active patient, uploaded image, and prediction in
# session_state so Dashboard -> History -> Dashboard preserves
# the current screening instead of starting over.

DEFAULT_SESSION_STATE = {
    "saved_screening_key": None,
    "uploaded_image_bytes": None,
    "uploaded_image_name": None,
    "uploaded_image_type": None,
    "uploaded_image_key": None,
    "prediction_result": None,
    "image_quality_result": None,
    "patient_name_saved": "",
    "patient_id_saved": "",
    "patient_age_saved": 30,
    "gender_saved": "Male",
    "screening_notes_saved": "",
}

for _state_key, _default_value in DEFAULT_SESSION_STATE.items():
    if _state_key not in st.session_state:
        st.session_state[_state_key] = _default_value


def _save_patient_name():
    st.session_state.patient_name_saved = st.session_state.patient_name_widget


def _save_patient_id():
    st.session_state.patient_id_saved = st.session_state.patient_id_widget


def _save_patient_age():
    st.session_state.patient_age_saved = st.session_state.patient_age_widget


def _save_gender():
    st.session_state.gender_saved = st.session_state.gender_widget


def _save_screening_notes():
    st.session_state.screening_notes_saved = st.session_state.screening_notes_widget

# ============================================================
# PATIENT INFORMATION VALIDATION
# ============================================================

def validate_patient_information(
    patient_name,
    patient_id,
    patient_age,
    gender
):
    """Validate patient information before starting screening."""

    errors = []

    # Patient name
    cleaned_name = patient_name.strip()

    if not cleaned_name:
        errors.append("Patient name is required.")
    elif len(cleaned_name) < 2:
        errors.append("Patient name must contain at least 2 characters.")
    elif len(cleaned_name) > 100:
        errors.append("Patient name must be 100 characters or fewer.")
    elif not any(character.isalpha() for character in cleaned_name):
        errors.append("Patient name must contain at least one letter.")
    elif any(character.isdigit() for character in cleaned_name):
        errors.append("Patient name cannot contain numbers.")

    # Patient ID
    cleaned_id = patient_id.strip()

    if not cleaned_id:
        errors.append("Patient ID is required.")
    elif len(cleaned_id) > 50:
        errors.append("Patient ID must be 50 characters or fewer.")
    elif not re.fullmatch(r"[A-Za-z0-9_-]+", cleaned_id):
        errors.append(
            "Patient ID can contain letters, numbers, hyphens, "
            "and underscores only."
        )

    # Age
    try:
        numeric_age = int(patient_age)
    except (TypeError, ValueError):
        numeric_age = None

    if numeric_age is None:
        errors.append("Please enter a valid age.")
    elif numeric_age < 1 or numeric_age > 120:
        errors.append("Age must be between 1 and 120 years.")

    # Gender
    allowed_genders = [
        "Male",
        "Female",
        "Other",
        "Prefer not to say",
    ]

    if gender not in allowed_genders:
        errors.append("Please select a valid gender option.")

    return errors


# ============================================================
# IMAGE QUALITY VALIDATION
# ============================================================

def validate_retina_image_quality(image):
    """Run conservative pre-screening quality checks on the uploaded image.

    These checks are intentionally designed to reject only obviously unusable
    images. They do not determine diabetic-retinopathy severity.
    """
    checks = []
    width, height = image.size

    # 1. Minimum usable dimensions.
    if width < 128 or height < 128:
        checks.append(
            f"Image resolution is too small ({width}×{height}). "
            "Please upload a higher-resolution retinal fundus image."
        )

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = np.mean(rgb, axis=2)

    # 2. Reject blank / nearly blank images.
    gray_std = float(np.std(gray))
    gray_mean = float(np.mean(gray))

    if gray_std < 5.0:
        checks.append(
            "The image appears blank or nearly uniform. "
            "Please upload a clear retinal fundus image."
        )
    elif gray_mean < 5.0:
        checks.append(
            "The image is almost completely black. "
            "Please upload a visible retinal fundus image."
        )
    elif gray_mean > 250.0:
        checks.append(
            "The image is almost completely white. "
            "Please upload a visible retinal fundus image."
        )

    # 3. Reject extremely blurry images using a lightweight edge-variation
    # metric. This is deliberately conservative so normal fundus images pass.
    horizontal_diff = np.diff(gray, axis=1)
    vertical_diff = np.diff(gray, axis=0)
    edge_variation = float(
        (np.var(horizontal_diff) + np.var(vertical_diff)) / 2.0
    )

    if edge_variation < 1.5:
        checks.append(
            "The image appears extremely blurry or lacks visible detail. "
            "Please upload a sharper retinal fundus image."
        )

    # 4. Check that there is enough visible image content.
    # Fundus photos commonly contain a dark background, so this threshold is
    # intentionally lenient and is not based on a simple black-pixel ratio.
    visible_ratio = float(np.mean(gray > 12.0))

    if visible_ratio < 0.08:
        checks.append(
            "Very little retinal image content is visible. "
            "Please upload a properly framed fundus photograph."
        )

    # 5. Basic color/content sanity check for the expected fundus-photo format.
    channel_spread = float(np.mean(np.max(rgb, axis=2) - np.min(rgb, axis=2)))
    if channel_spread < 2.0 and gray_std < 18.0:
        checks.append(
            "The uploaded image does not contain enough visual variation "
            "for reliable retinal screening."
        )

    return {
        "valid": len(checks) == 0,
        "issues": checks,
        "width": width,
        "height": height,
        "gray_std": gray_std,
        "edge_variation": edge_variation,
        "visible_ratio": visible_ratio,
    }


# ============================================================
# SCREENING HISTORY PAGE
# ============================================================

def show_screening_history():
    """Display saved screening records from the FastAPI database."""

    st.markdown(
        """
        <div class="history-page-title">📚 Screening History</div>
        <div class="history-page-subtitle">
            Review and manage previous diabetic retinopathy screening results
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # LOAD HISTORY
    # --------------------------------------------------------

    refresh_col, _ = st.columns([1, 5])

    with refresh_col:
        if st.button(
            "🔄 Refresh History",
            use_container_width=True
        ):
            st.rerun()

    try:
        history_response = requests.get(
            SCREENINGS_URL,
            timeout=10
        )
        history_response.raise_for_status()

        screening_history = history_response.json()

        if isinstance(screening_history, dict):
            screening_history = (
                screening_history.get("screenings")
                or screening_history.get("data")
                or []
            )

    except requests.exceptions.RequestException as e:
        st.error(
            f"⚠️ Could not load screening history: {e}"
        )
        return

    if not screening_history:
        st.info(
            "📭 No screening records found. "
            "Complete a screening from the Dashboard to create your first history record."
        )
        return

    # --------------------------------------------------------
    # CALCULATE ANALYTICS
    # --------------------------------------------------------

    severity_order = [
        "No DR",
        "Mild",
        "Moderate",
        "Severe",
        "Proliferative DR"
    ]

    severity_counts = {
        severity: 0
        for severity in severity_order
    }

    for record in screening_history:
        label = record.get(
            "predicted_label",
            "Unknown"
        )

        if label in severity_counts:
            severity_counts[label] += 1

    total_screenings = len(screening_history)

    no_dr_cases = severity_counts["No DR"]

    dr_cases = total_screenings - no_dr_cases

    most_common_result = max(
        severity_counts,
        key=severity_counts.get
    )

    # --------------------------------------------------------
    # SUMMARY CARDS
    # --------------------------------------------------------

    card1, card2, card3, card4 = st.columns(4)

    with card1:
        st.metric(
            "Total Screenings",
            total_screenings
        )

    with card2:
        st.metric(
            "No DR",
            no_dr_cases
        )

    with card3:
        st.metric(
            "DR Cases",
            dr_cases
        )

    with card4:
        st.metric(
            "Most Common",
            most_common_result
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # SEVERITY ANALYTICS
    # --------------------------------------------------------

    st.markdown("### 📊 Severity Distribution")

    chart_col, breakdown_col = st.columns(
        [3, 2],
        gap="large"
    )

    with chart_col:

        chart_df = pd.DataFrame(
            {
                "Severity": severity_order,
                "Screenings": [
                    severity_counts[severity]
                    for severity in severity_order
                ]
            }
        )

        # Dark-theme chart so it matches the rest of the application.
        fig, ax = plt.subplots(
            figsize=(8, 4.2),
            facecolor="#11141b"
        )
        ax.set_facecolor("#11141b")

        bars = ax.bar(
            chart_df["Severity"],
            chart_df["Screenings"],
            color="#4f8cff",
            width=0.62
        )

        ax.set_title(
            "Screenings by Severity",
            fontsize=14,
            fontweight="bold",
            color="#f8fafc",
            pad=14
        )

        ax.set_ylabel(
            "Number of Screenings",
            color="#cbd5e1"
        )

        ax.set_xlabel(
            "Severity Level",
            color="#cbd5e1"
        )

        ax.tick_params(
            axis="x",
            rotation=20,
            colors="#cbd5e1"
        )

        ax.tick_params(
            axis="y",
            colors="#94a3b8"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")
        ax.grid(axis="y", alpha=0.12, color="#94a3b8")
        ax.set_axisbelow(True)

        # Show count above each bar.
        for bar in bars:
            height = bar.get_height()

            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.04,
                    str(int(height)),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#f8fafc"
                )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    with breakdown_col:

        st.markdown("#### Severity Breakdown")

        for severity in severity_order:

            count = severity_counts[severity]

            percentage = (
                (count / total_screenings) * 100
                if total_screenings
                else 0
            )

            st.markdown(
                f"**{severity}**  \n"
                f"{count} screening(s) • "
                f"{percentage:.1f}%"
            )

            st.progress(
                percentage / 100
            )

    # --------------------------------------------------------
    # HISTORY TABLE
    # --------------------------------------------------------

    st.markdown("### 📋 Recent Screenings")

    history_rows = []

    for record in screening_history:

        created_at = (
            record.get("created_at")
            or record.get("timestamp")
            or record.get("date")
            or "N/A"
        )

        confidence = record.get(
            "confidence"
        )

        try:
            confidence_display = (
                f"{float(confidence):.1%}"
                if confidence is not None
                else "N/A"
            )
        except (TypeError, ValueError):
            confidence_display = str(confidence)

        history_rows.append(
            {
                "Date": created_at,
                "Patient": (
                    record.get("patient_name")
                    or "Not provided"
                ),
                "Patient ID": (
                    record.get("patient_id")
                    or "Not provided"
                ),
                "Age": record.get(
                    "age",
                    "N/A"
                ),
                "Gender": record.get(
                    "gender",
                    "N/A"
                ),
                "Prediction": record.get(
                    "predicted_label",
                    "Unknown"
                ),
                "Confidence": confidence_display
            }
        )

    st.dataframe(
        pd.DataFrame(history_rows),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # MANAGE SCREENING
    # --------------------------------------------------------

    st.markdown("### 🔎 Manage Screening")

    record_options = {}

    for record in screening_history:

        screening_id = (
            record.get("id")
            or record.get("screening_id")
        )

        if screening_id is None:
            continue

        patient_name = (
            record.get("patient_name")
            or "Unnamed Patient"
        )

        patient_id = (
            record.get("patient_id")
            or "No ID"
        )

        prediction = (
            record.get("predicted_label")
            or "Unknown"
        )

        option_label = (
            f"#{screening_id} | "
            f"{patient_name} | "
            f"{patient_id} | "
            f"{prediction}"
        )

        record_options[option_label] = screening_id

    if record_options:

        selected_option = st.selectbox(
            "Select a screening record",
            list(record_options.keys())
        )

        selected_id = record_options[
            selected_option
        ]

        view_col, delete_col = st.columns(2)

        with view_col:
            view_record = st.button(
                "👁️ View Details",
                use_container_width=True
            )

        with delete_col:
            delete_record = st.button(
                "🗑️ Delete Screening",
                use_container_width=True
            )

        if view_record:

            try:
                detail_response = requests.get(
                    f"{SCREENINGS_URL}/{selected_id}",
                    timeout=10
                )

                detail_response.raise_for_status()

                detail = detail_response.json()

                st.markdown(
                    "#### 📋 Screening Details"
                )

                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:

                    st.write(
                        "**Patient Name:**",
                        detail.get(
                            "patient_name",
                            "Not provided"
                        )
                    )

                    st.write(
                        "**Patient ID:**",
                        detail.get(
                            "patient_id",
                            "Not provided"
                        )
                    )

                    st.write(
                        "**Age:**",
                        detail.get(
                            "age",
                            "N/A"
                        )
                    )

                    st.write(
                        "**Gender:**",
                        detail.get(
                            "gender",
                            "N/A"
                        )
                    )

                with detail_col2:

                    st.write(
                        "**Prediction:**",
                        detail.get(
                            "predicted_label",
                            "Unknown"
                        )
                    )

                    confidence = detail.get(
                        "confidence"
                    )

                    try:
                        confidence = (
                            f"{float(confidence):.1%}"
                            if confidence is not None
                            else "N/A"
                        )
                    except (TypeError, ValueError):
                        pass

                    st.write(
                        "**Confidence:**",
                        confidence
                    )

                    st.write(
                        "**Date:**",
                        detail.get(
                            "created_at",
                            detail.get(
                                "timestamp",
                                "N/A"
                            )
                        )
                    )

                notes = detail.get(
                    "screening_notes",
                    ""
                )

                if notes:
                    st.write(
                        "**Screening Notes:**"
                    )
                    st.info(notes)

            except requests.exceptions.RequestException as e:

                st.error(
                    f"❌ Could not load screening details: {e}"
                )

        if delete_record:

            try:
                delete_response = requests.delete(
                    f"{SCREENINGS_URL}/{selected_id}",
                    timeout=10
                )

                delete_response.raise_for_status()

                st.success(
                    "✅ Screening deleted successfully."
                )

                st.rerun()

            except requests.exceptions.RequestException as e:

                st.error(
                    f"❌ Could not delete screening: {e}"
                )

# ============================================================
# END SCREENING HISTORY
# ============================================================


# ============================================================
# CUSTOM STYLING — FUTURISTIC MEDICAL UI
# ============================================================

st.markdown(
    r"""
    <style>
    :root {
        --bg: #050a14;
        --panel: #0b1424;
        --panel-2: #0e1a2d;
        --line: rgba(76, 176, 255, 0.28);
        --cyan: #31d8ff;
        --blue: #3b82ff;
        --purple: #9b6cff;
        --text: #f4f7ff;
        --muted: #91a4bf;
    }

    .stApp {
        background:
            radial-gradient(circle at 82% 34%, rgba(40, 107, 255, 0.10), transparent 28%),
            radial-gradient(circle at 12% 70%, rgba(0, 211, 255, 0.055), transparent 25%),
            linear-gradient(180deg, #050913 0%, #070c16 55%, #050912 100%);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: .17;
        background-image:
            linear-gradient(rgba(49,216,255,.13) 1px, transparent 1px),
            linear-gradient(90deg, rgba(49,216,255,.13) 1px, transparent 1px);
        background-size: 32px 32px;
        mask-image: linear-gradient(to right, black, transparent 72%);
        z-index: 0;
    }

    .block-container {
        position: relative;
        z-index: 1;
        max-width: 1500px;
        padding-top: 1.25rem;
        padding-bottom: 4rem;
    }

    /* ---------------- TOP NAV ---------------- */
    .top-nav {
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 22px;
        padding: 18px 24px;
        margin: 2px 0 18px;
        border: 1px solid rgba(67, 165, 255, .32);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(12,27,49,.94), rgba(9,18,34,.92));
        box-shadow: 0 0 0 1px rgba(255,255,255,.025) inset, 0 16px 45px rgba(0,0,0,.35), 0 0 28px rgba(40,150,255,.08);
    }
    .top-nav::after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        right: -80px;
        top: -120px;
        background: radial-gradient(circle, rgba(50,174,255,.16), transparent 67%);
        pointer-events: none;
    }
    .top-nav-brand { display:flex; align-items:center; gap:14px; }
    .top-nav-logo {
        width: 48px; height: 48px; border-radius: 15px;
        display:flex; align-items:center; justify-content:center;
        background: linear-gradient(145deg,#0d6eff,#2455ff 55%,#7a4dff);
        border: 1px solid rgba(111,215,255,.55);
        box-shadow: 0 0 22px rgba(39,126,255,.35), inset 0 1px 0 rgba(255,255,255,.18);
        font-size: 24px;
    }
    .top-nav-title { color:#f6f9ff; font-size:20px; font-weight:800; }
    .top-nav-subtitle { color:#91a4bf; font-size:12px; margin-top:3px; }
    .top-nav-right {
        display:flex; align-items:center; justify-content:flex-end; gap:14px; flex-wrap:wrap;
    }
    .developer-credit-top {
        display:flex; align-items:center; gap:8px; color:#8fa4bf; font-size:10px; white-space:nowrap;
        padding-left:10px; border-left:1px solid rgba(96,165,250,0.18);
    }
    .developer-credit-top b { color:#dbeafe; font-weight:750; }
    .developer-credit-top img {
        width:28px; height:28px; object-fit:cover; border-radius:50%;
        border:1px solid rgba(147,197,253,0.55); box-shadow:0 0 12px rgba(59,130,246,0.16);
        flex:0 0 28px;
    }
    .developer-footer { margin-top:34px; padding:0 4px 18px 4px; }
    .developer-footer-line {
        height:1px; margin-bottom:13px;
        background:linear-gradient(90deg, transparent, rgba(96,165,250,0.28), rgba(139,125,255,0.24), transparent);
    }
    .developer-footer-content {
        display:flex; align-items:center; justify-content:center; gap:8px;
        color:#71849d; font-size:11px; letter-spacing:0.1px;
    }
    .developer-footer-text b { color:#a9c7e8; font-weight:700; }
    .developer-footer-content img {
        width:24px; height:24px; object-fit:cover; border-radius:50%;
        border:1px solid rgba(147,197,253,0.48); box-shadow:0 0 10px rgba(59,130,246,0.12);
    }
    .top-nav-model {
        display:flex; align-items:center; gap:9px; white-space:nowrap;
        padding: 10px 15px; border-radius: 999px;
        background: rgba(16,35,61,.75); border:1px solid rgba(75,165,255,.2);
        color:#a9bad2; font-size:12px;
    }
    .top-nav-model span:first-child { color:#c9eaff; font-weight:800; }
    .top-nav-model::before {
        content:""; width:9px; height:9px; border-radius:50%;
        background:#16e6a0; box-shadow:0 0 12px rgba(22,230,160,.9);
        animation: pulseDot 2s infinite;
    }
    @keyframes pulseDot { 50% { opacity:.45; transform:scale(.82); } }

    /* ---------------- NAVIGATION ---------------- */
    div[data-testid="stRadio"] { margin: 0 0 22px 0; }
    div[data-testid="stRadio"] > label { display:none; }
    div[data-testid="stRadio"] > div {
        width: fit-content; min-width: 500px; gap: 8px;
        background: rgba(10,21,38,.88);
        border:1px solid rgba(76,176,255,.22);
        border-radius:16px; padding:6px;
        box-shadow: 0 12px 30px rgba(0,0,0,.22);
    }
    div[data-testid="stRadio"] label {
        border:1px solid transparent !important; border-radius:11px !important;
        padding:10px 25px !important; color:#9fb1c9 !important;
        background:transparent !important; transition:all .2s ease;
    }
    div[data-testid="stRadio"] label:hover {
        color:#fff !important; background:rgba(31,110,214,.15) !important;
        border-color:rgba(49,216,255,.2) !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        color:#fff !important;
        background:linear-gradient(135deg,#116cf4,#365cff 60%,#6449e8) !important;
        border-color:rgba(91,204,255,.8) !important;
        box-shadow:0 0 20px rgba(39,132,255,.42), inset 0 1px 0 rgba(255,255,255,.2);
    }

    /* ---------------- HERO ---------------- */
    .hero-wrap {
        position:relative; text-align:center; padding:28px 10px 18px; margin-bottom:12px; overflow:hidden;
    }
    .hero-retina {
        position:absolute; right:-30px; top:10px; width:270px; height:270px; border-radius:50%; opacity:.23;
        background:
          radial-gradient(circle at 50% 50%, rgba(51,150,255,.45) 0 2%, transparent 3%),
          radial-gradient(circle at 50% 50%, transparent 0 25%, rgba(50,160,255,.28) 26% 27%, transparent 28% 42%, rgba(50,160,255,.18) 43% 44%, transparent 45%),
          radial-gradient(circle, rgba(18,80,180,.32), transparent 64%);
        filter: blur(.2px); pointer-events:none;
    }
    .hero-title {
        position:relative; display:inline-block; font-size:52px; line-height:1.08; font-weight:900; letter-spacing:-1.8px;
        background:linear-gradient(90deg,#f7fbff 0%,#d8e6ff 20%,#43d7ff 42%,#8b7dff 63%,#e88cff 82%,#f7fbff 100%);
        background-size:240% auto; -webkit-background-clip:text; background-clip:text; color:transparent;
        animation: titleFlow 7s linear infinite;
        text-shadow:0 0 24px rgba(77,172,255,.12);
    }
    @keyframes titleFlow { to { background-position:240% center; } }
    .hero-subtitle { color:#8fa5c2; font-size:16px; margin-top:10px; }

    /* ---------------- WORKFLOW ---------------- */
    .workflow {
        display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:4px auto 25px; max-width:1150px;
    }
    .workflow-item {
        display:flex; align-items:center; gap:10px; min-width:0; padding:10px 12px;
        border:1px solid rgba(65,153,236,.16); border-radius:15px; background:rgba(9,22,40,.62);
    }
    .workflow-icon {
        flex:0 0 42px; width:42px; height:42px; border-radius:50%; display:flex; align-items:center; justify-content:center;
        color:#65ddff; border:1px solid rgba(63,191,255,.5); background:rgba(20,75,130,.32);
        box-shadow:0 0 15px rgba(35,152,255,.16);
    }
    .workflow-title { color:#f1f6ff; font-weight:750; font-size:13px; }
    .workflow-text { color:#8198b6; font-size:11px; margin-top:2px; }

    /* ---------------- SECTION CARDS ---------------- */
    .section-card {
        position:relative; overflow:hidden; border:1px solid rgba(63,163,255,.34); border-radius:24px;
        background:linear-gradient(145deg,rgba(12,29,51,.92),rgba(7,17,31,.92));
        box-shadow:0 18px 48px rgba(0,0,0,.34), 0 0 25px rgba(31,130,255,.08), inset 0 1px 0 rgba(255,255,255,.035);
        padding:22px 28px 25px; margin:10px 0 22px;
    }
    .section-card::before {
        content:""; position:absolute; inset:0; pointer-events:none;
        background:radial-gradient(circle at 90% 0%,rgba(57,150,255,.12),transparent 32%);
    }
    .section-heading { display:flex; align-items:center; gap:13px; margin-bottom:18px; }
    .section-heading-icon {
        width:50px; height:50px; border-radius:14px; display:flex; align-items:center; justify-content:center;
        background:linear-gradient(145deg,#1176ff,#324ff0 60%,#7049e8);
        border:1px solid rgba(91,214,255,.52); box-shadow:0 0 22px rgba(35,130,255,.27);
    }
    .section-heading h2 { margin:0; font-size:25px; color:#f4f8ff; }
    .section-heading p { margin:3px 0 0; color:#71bff2; font-size:13px; }
    .security-pill { margin-left:auto; padding:10px 16px; border-radius:13px; border:1px solid rgba(49,216,255,.28); background:rgba(18,71,112,.3); color:#cfeeff; font-size:12px; }

    /* Inputs */
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background:rgba(14,28,49,.82) !important; border:1px solid rgba(93,157,218,.26) !important;
        color:#edf5ff !important; border-radius:11px !important;
    }
    div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus {
        border-color:#2fd8ff !important; box-shadow:0 0 0 1px #2fd8ff, 0 0 18px rgba(47,216,255,.16) !important;
    }
    div[data-testid="stFileUploader"] section {
        background:rgba(10,24,42,.72) !important; border:1px dashed rgba(69,182,255,.42) !important; border-radius:17px !important;
        box-shadow:0 0 20px rgba(29,137,255,.07);
    }

    /* Buttons */
    .stButton > button {
        border-radius:11px; border:1px solid rgba(72,181,255,.36);
        background:linear-gradient(135deg,rgba(17,83,173,.92),rgba(70,64,178,.92));
        color:#fff; font-weight:700; transition:.18s ease;
        box-shadow:0 7px 20px rgba(26,91,200,.18);
    }
    .stButton > button:hover { transform:translateY(-2px); border-color:#58dfff; box-shadow:0 0 20px rgba(46,193,255,.28); }

    .main-title, .subtitle { display:none; }

    /* History */
    .history-page-title { font-size:36px; font-weight:850; color:#f4f8ff; margin:10px 0 3px; }
    .history-page-subtitle { color:#8fa5c2; font-size:15px; margin-bottom:22px; }

    /* Image quality validation */
    .quality-warning-card {
        border:1px solid rgba(255,180,70,.45);
        border-radius:18px;
        padding:18px 20px;
        margin:16px 0 12px;
        background:linear-gradient(145deg,rgba(73,48,18,.34),rgba(34,24,12,.42));
        box-shadow:0 12px 28px rgba(0,0,0,.24);
    }
    .quality-warning-title { color:#ffd28a; font-size:18px; font-weight:800; }
    .quality-warning-subtitle { color:#b9a98f; font-size:13px; margin-top:5px; }

    /* Visual explanation */
    .visual-section-header { margin:10px 0 22px; }
    .visual-section-title { font-size:1.65rem; font-weight:800; }
    .visual-section-subtitle { color:#8fa5c2; margin-top:5px; font-size:.95rem; }
    .visual-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:22px; align-items:stretch; }
    .visual-card { background:linear-gradient(145deg,#0e1a2d,#091321); border:1px solid rgba(76,176,255,.25); border-radius:18px; padding:15px; box-shadow:0 14px 32px rgba(0,0,0,.28); transition:.22s ease; }
    .visual-card:hover { transform:translateY(-4px); border-color:rgba(73,210,255,.65); box-shadow:0 0 25px rgba(31,160,255,.15),0 18px 38px rgba(0,0,0,.3); }
    .visual-image-frame { overflow:hidden; border-radius:12px; border:1px solid rgba(255,255,255,.2); }
    .visual-image { width:100%; display:block; transition:transform .3s ease; }
    .visual-card:hover .visual-image { transform:scale(1.025); }
    .visual-hover-label { margin-top:11px; color:#dcecff; font-weight:750; font-size:14px; }
    .visual-disclaimer { color:#7f96b4; font-size:11px; margin-top:3px; }

    @media (max-width: 900px) {
        .top-nav { flex-direction:column; align-items:flex-start; }
        .top-nav-right { width:100%; justify-content:space-between; }
        .developer-credit-top { border-left:none; padding-left:0; }
        .top-nav-model { white-space:normal; }
        .hero-title { font-size:38px; }
        .workflow { grid-template-columns:repeat(2,1fr); }
        .visual-grid { grid-template-columns:1fr; }
        div[data-testid="stRadio"] > div { min-width:0; width:100%; }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PDF REPORT HELPERS
# ============================================================

def _safe_filename(value):
    """Create a safe filename component."""
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    return value.strip("_") or "patient"


def _pil_to_png_bytes(pil_image):
    """Convert a PIL image to PNG bytes for the PDF."""
    buffer = io.BytesIO()
    pil_image.convert("RGB").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def _pdf_image_from_pil(pil_image, max_width=3.15 * inch, max_height=3.0 * inch):
    """Create a reportlab image while preserving the original aspect ratio."""
    image_bytes = _pil_to_png_bytes(pil_image)
    image_stream = io.BytesIO(image_bytes)

    width, height = pil_image.size
    if width <= 0 or height <= 0:
        return None

    scale = min(max_width / width, max_height / height)
    return PDFImage(
        image_stream,
        width=width * scale,
        height=height * scale,
    )


def create_screening_pdf(
    patient_name,
    patient_id,
    patient_age,
    gender,
    screening_notes,
    predicted_label,
    current_result,
    confidence,
    normalized_probabilities,
    original_image,
    heatmap_image=None,
):
    """Build the complete diabetic-retinopathy screening PDF in memory."""

    pdf_buffer = io.BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        title="Diabetic Retinopathy Screening Report",
        author="DR Screening Application",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=6,
    )

    small_style = ParagraphStyle(
        "ReportSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#555555"),
    )

    result_style = ParagraphStyle(
        "ReportResult",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#123B63"),
        spaceAfter=5,
    )

    story = []

    generated_at = datetime.now().strftime("%d %B %Y, %I:%M %p")

    # Header
    story.append(
        Paragraph(
            "DIABETIC RETINOPATHY SCREENING REPORT",
            title_style,
        )
    )
    story.append(
        Paragraph(
            f"AI-assisted retinal image screening report | Generated: {escape(generated_at)}",
            subtitle_style,
        )
    )

    # Patient information
    story.append(Paragraph("1. Patient Information", heading_style))

    patient_data = [
        ["Patient Name", patient_name.strip() if patient_name.strip() else "Not provided"],
        ["Patient ID", patient_id.strip() if patient_id.strip() else "Not provided"],
        ["Age", f"{patient_age} years"],
        ["Gender", gender],
    ]

    patient_table = Table(
        patient_data,
        colWidths=[1.45 * inch, 4.85 * inch],
    )
    patient_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#123B63")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B8C7D9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(patient_table)

    if screening_notes.strip():
        story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                f"<b>Screening Notes:</b> {escape(screening_notes.strip())}",
                body_style,
            )
        )

    # Result
    story.append(Paragraph("2. Screening Result", heading_style))
    story.append(
        Paragraph(
            f"Prediction: {escape(current_result['name'])}",
            result_style,
        )
    )
    story.append(
        Paragraph(
            f"Confidence: {confidence:.1%}",
            body_style,
        )
    )
    story.append(
        Paragraph(
            f"<b>Status:</b> {escape(current_result['status'])}",
            body_style,
        )
    )
    story.append(
        Paragraph(
            escape(current_result["description"]),
            body_style,
        )
    )
    story.append(
        Paragraph(
            f"<b>Recommendation:</b> {escape(current_result['recommendation'])}",
            body_style,
        )
    )

    # Images
    story.append(Paragraph("3. Retinal Images", heading_style))

    original_pdf_image = _pdf_image_from_pil(original_image)

    if heatmap_image is not None:
        heatmap_pdf_image = _pdf_image_from_pil(heatmap_image)
    else:
        heatmap_pdf_image = None

    image_cells = []
    image_headers = []

    if original_pdf_image is not None:
        image_cells.append(original_pdf_image)
        image_headers.append(
            Paragraph("<b>Original Retina</b>", body_style)
        )

    if heatmap_pdf_image is not None:
        image_cells.append(heatmap_pdf_image)
        image_headers.append(
            Paragraph("<b>Grad-CAM Attention</b>", body_style)
        )

    if image_cells:
        image_table = Table(
            [image_headers, image_cells],
            colWidths=[3.15 * inch] * len(image_cells),
        )
        image_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D1DC")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D1DC")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(image_table)
    else:
        story.append(
            Paragraph(
                "Retinal images were not available for inclusion in the report.",
                small_style,
            )
        )

    # Probability distribution
    story.append(Paragraph("4. Probability by Severity", heading_style))

    probability_rows = [["Severity", "Probability"]]
    for label in severity_levels:
        probability_rows.append(
            [
                label,
                f"{normalized_probabilities[label]:.1%}",
            ]
        )

    probability_table = Table(
        probability_rows,
        colWidths=[4.5 * inch, 1.8 * inch],
    )
    probability_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B63")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B8C7D9")),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(probability_table)

    # Interpretation
    story.append(Paragraph("5. Screening Interpretation", heading_style))
    story.append(
        Paragraph(
            escape(current_result["description"]),
            body_style,
        )
    )
    story.append(
        Paragraph(
            "This interpretation is generated from the model prediction and is not a clinical diagnosis.",
            small_style,
        )
    )

    # Model explanation
    story.append(Paragraph("6. Model Explanation", heading_style))
    story.append(
        Paragraph(
            "The Grad-CAM visualization highlights retinal image regions "
            "that contributed more strongly to the model's prediction. "
            "It is provided for model interpretability and should not be "
            "considered a clinical explanation.",
            body_style,
        )
    )

    # Model performance
    story.append(Paragraph("7. Model Performance", heading_style))
    performance_data = [
        ["Metric", "Value"],
        ["Test Accuracy", "84.91%"],
        ["Quadratic Weighted Kappa", "0.9005"],
    ]

    performance_table = Table(
        performance_data,
        colWidths=[4.5 * inch, 1.8 * inch],
    )
    performance_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B63")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B8C7D9")),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(performance_table)
    story.append(
        Spacer(1, 5)
    )
    story.append(
        Paragraph(
            "Performance measured on the held-out test set of 550 retinal images using DR Model V3.",
            small_style,
        )
    )

    # Disclaimer
    story.append(Paragraph("8. Important Disclaimer", heading_style))
    story.append(
        Paragraph(
            "<b>This application is an AI-assisted educational screening demo "
            "and is not a medical diagnostic tool.</b>",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Results should not be used as a substitute for evaluation by a "
            "qualified eye-care professional. A healthcare professional should "
            "make the final clinical assessment.",
            body_style,
        )
    )

    document.build(story)
    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()



# ============================================================
# PROFESSIONAL TOP NAVIGATION
# ============================================================

developer_image_uri = get_developer_image_uri()

st.markdown(
    fr"""
    <div class="top-nav">
        <div class="top-nav-brand">
            <div class="top-nav-logo">👁️</div>
            <div>
                <div class="top-nav-title">DR Screening</div>
                <div class="top-nav-subtitle">AI-assisted diabetic retinopathy screening</div>
            </div>
        </div>
        <div class="top-nav-right">
            <div class="top-nav-model">
                <span>EfficientNet-B0</span><span>•</span><span>Model V3</span><span>•</span><span>84.91% Accuracy</span>
            </div>
            <div class="developer-credit-top">
                <span>Developed by <b>Arpit Pandey</b></span>
                <img src="{developer_image_uri}" alt="Arpit Pandey" />
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

selected_page = st.radio(
    "Application navigation",
    ["🏠 Dashboard", "📚 Screening History"],
    horizontal=True,
    label_visibility="collapsed",
)

if selected_page == "📚 Screening History":
    show_screening_history()
    st.stop()

st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    r"""
    <div class="hero-wrap">
        <div class="hero-retina"></div>
        <div>
            <span class="hero-title">Diabetic Retinopathy Screening</span>
        </div>
        <div class="hero-subtitle">AI-assisted screening of retinal fundus images with Grad-CAM explainability</div>
    </div>

    <div class="workflow">
        <div class="workflow-item"><div class="workflow-icon">🖼️</div><div><div class="workflow-title">1. Upload Image</div><div class="workflow-text">Retinal fundus image</div></div></div>
        <div class="workflow-item"><div class="workflow-icon">🧠</div><div><div class="workflow-title">2. AI Analysis</div><div class="workflow-text">Deep learning model</div></div></div>
        <div class="workflow-item"><div class="workflow-icon">📊</div><div><div class="workflow-title">3. DR Severity</div><div class="workflow-text">5-class classification</div></div></div>
        <div class="workflow-item"><div class="workflow-icon">🎯</div><div><div class="workflow-title">4. Explainability</div><div class="workflow-text">Grad-CAM visualization</div></div></div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown(
    r"""
    <div class="section-card">
        <div class="section-heading">
            <div class="section-heading-icon">👤</div>
            <div>
                <h2>Patient Information</h2>
                <p>Enter patient details to begin screening</p>
            </div>
            <div class="security-pill">🛡️ <b>Secure &amp; Confidential</b><br><span style="opacity:.75">Local screening data</span></div>
        </div>
    """,
    unsafe_allow_html=True
)

patient_col1, patient_col2, patient_col3 = st.columns(3)

with patient_col1:
    patient_name = st.text_input(
        "Patient Name",
        value=st.session_state.patient_name_saved,
        placeholder="Enter patient name",
        key="patient_name_widget",
        on_change=_save_patient_name
    )
    st.session_state.patient_name_saved = patient_name

with patient_col2:
    patient_id = st.text_input(
        "Patient ID",
        value=st.session_state.patient_id_saved,
        placeholder="Enter patient ID",
        key="patient_id_widget",
        on_change=_save_patient_id
    )
    st.session_state.patient_id_saved = patient_id

with patient_col3:
    patient_age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=int(st.session_state.patient_age_saved),
        step=1,
        key="patient_age_widget",
        on_change=_save_patient_age
    )
    st.session_state.patient_age_saved = int(patient_age)

gender_options = [
    "Male",
    "Female",
    "Other",
    "Prefer not to say"
]

gender = st.selectbox(
    "Gender",
    gender_options,
    index=gender_options.index(st.session_state.gender_saved),
    key="gender_widget",
    on_change=_save_gender
)
st.session_state.gender_saved = gender

screening_notes = st.text_area(
    "Screening Notes",
    value=st.session_state.screening_notes_saved,
    placeholder="Optional notes about this screening...",
    key="screening_notes_widget",
    on_change=_save_screening_notes
)
st.session_state.screening_notes_saved = screening_notes

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# VALIDATE PATIENT INFORMATION
# ============================================================

patient_validation_errors = validate_patient_information(
    patient_name,
    patient_id,
    patient_age,
    gender
)

if patient_validation_errors:
    st.warning(
        "⚠️ Please complete the required patient information "
        "before starting the screening."
    )

    for validation_error in patient_validation_errors:
        st.error(f"• {validation_error}")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📤 Upload Retina Image")

uploaded_file = st.file_uploader(
    "Choose a retinal fundus image",
    type=["png", "jpg", "jpeg"],
    key="retina_uploader"
)

if patient_validation_errors:
    st.info("Enter valid patient details above to continue.")
    st.stop()


# ============================================================
# RESTORE ACTIVE SCREENING AFTER NAVIGATION
# ============================================================
# When History is opened, the Dashboard widgets are not rendered.
# Therefore the uploaded file/result must come from session_state
# when the user returns to Dashboard.

active_image_bytes = None
active_image_name = None
active_image_type = None

if uploaded_file is not None:
    new_image_bytes = uploaded_file.getvalue()
    new_image_key = hashlib.sha256(new_image_bytes).hexdigest()

    if st.session_state.uploaded_image_key != new_image_key:
        st.session_state.uploaded_image_bytes = new_image_bytes
        st.session_state.uploaded_image_name = uploaded_file.name
        st.session_state.uploaded_image_type = uploaded_file.type
        st.session_state.uploaded_image_key = new_image_key
        st.session_state.prediction_result = None
        st.session_state.image_quality_result = None
        st.session_state.saved_screening_key = None

    active_image_bytes = st.session_state.uploaded_image_bytes
    active_image_name = st.session_state.uploaded_image_name
    active_image_type = st.session_state.uploaded_image_type

elif st.session_state.uploaded_image_bytes is not None:
    active_image_bytes = st.session_state.uploaded_image_bytes
    active_image_name = st.session_state.uploaded_image_name
    active_image_type = st.session_state.uploaded_image_type



# PROFESSIONAL TOP NAVIGATION
# ============================================================

st.markdown(
    """
    <style>

    /* Hide Streamlit's default sidebar. */
    section[data-testid="stSidebar"] {
        display: none;
    }

    /* Give the main content a cleaner, wider presentation. */
    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Top brand/navigation area. */
    .app-nav {
        background: linear-gradient(135deg, #151922 0%, #1b2130 100%);
        border: 1px solid #303746;
        border-radius: 18px;
        padding: 16px 22px 10px 22px;
        margin-bottom: 28px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.20);
    }

    .app-brand {
        font-size: 22px;
        font-weight: 750;
        letter-spacing: 0.2px;
        margin-bottom: 2px;
    }

    .app-brand-subtitle {
        color: #9ca3af;
        font-size: 13px;
        margin-bottom: 8px;
    }

    /* Style the horizontal Streamlit radio as a segmented navigation bar. */
    div[data-testid="stRadio"] > label {
        display: none;
    }

    div[data-testid="stRadio"] > div {
        gap: 8px;
    }

    div[data-testid="stRadio"] label {
        background: #202532;
        border: 1px solid #343b4a;
        border-radius: 10px;
        padding: 9px 18px;
        transition: all 0.15s ease;
    }

    div[data-testid="stRadio"] label:hover {
        border-color: #6d5dfc;
        background: #272d3b;
    }

    /* Cards used by the history page. */
    .history-page-title {
        font-size: 34px;
        font-weight: 750;
        margin: 4px 0 4px 0;
    }

    .history-page-subtitle {
        color: #9ca3af;
        font-size: 15px;
        margin-bottom: 22px;
    }



    /* ========================================================
       VISUAL EXPLANATION - IMAGE CARDS
       ======================================================== */

    .visual-section-header {
        margin: 8px 0 22px 0;
    }

    .visual-section-title {
        font-size: 1.65rem;
        font-weight: 750;
        letter-spacing: -0.02em;
    }

    .visual-section-subtitle {
        color: #9ca3af;
        margin-top: 5px;
        font-size: 0.95rem;
    }

    .visual-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 22px;
        align-items: stretch;
    }

    .visual-card {
        background: linear-gradient(145deg, #151a27, #10141e);
        border: 1px solid #2d3748;
        border-radius: 18px;
        padding: 18px;
        min-width: 0;
        box-shadow: 0 8px 26px rgba(0, 0, 0, 0.20);
        transition: transform 0.25s ease, border-color 0.25s ease,
                    box-shadow 0.25s ease;
    }

    .visual-card:hover {
        transform: translateY(-4px);
        border-color: #667eea;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.32);
    }

    .visual-card-heading {
        display: flex;
        align-items: center;
        gap: 11px;
        min-height: 52px;
        margin-bottom: 14px;
    }

    .visual-card-icon {
        font-size: 1.45rem;
        line-height: 1;
    }

    .visual-card-title {
        font-size: 1.12rem;
        font-weight: 750;
        line-height: 1.25;
    }

    .visual-card-subtitle {
        color: #9ca3af;
        font-size: 0.82rem;
        margin-top: 3px;
    }

    .visual-image-frame {
        position: relative;
        width: 100%;
        overflow: hidden;
        border: 2px solid rgba(255, 255, 255, 0.88);
        border-radius: 13px;
        background: #080b12;
        box-sizing: border-box;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.08),
                    0 8px 20px rgba(0,0,0,0.28);
    }

    .visual-image {
        display: block;
        width: 100%;
        height: 320px;
        object-fit: contain;
        background: #080b12;
        transition: transform 0.35s ease, filter 0.35s ease;
        transform-origin: center center;
        cursor: zoom-in;
    }

    .visual-image-frame:hover .visual-image {
        transform: scale(1.075);
        filter: brightness(1.06) saturate(1.04);
    }

    .visual-hover-label {
        position: absolute;
        right: 10px;
        bottom: 10px;
        padding: 5px 9px;
        border-radius: 999px;
        background: rgba(7, 10, 17, 0.76);
        border: 1px solid rgba(255,255,255,0.18);
        color: #f8fafc;
        font-size: 0.72rem;
        opacity: 0;
        transform: translateY(4px);
        transition: opacity 0.25s ease, transform 0.25s ease;
        pointer-events: none;
    }

    .visual-image-frame:hover .visual-hover-label {
        opacity: 1;
        transform: translateY(0);
    }

    .visual-card-description {
        color: #b7c0ce;
        font-size: 0.86rem;
        line-height: 1.55;
        margin-top: 13px;
        min-height: 42px;
    }

    .visual-tip {
        text-align: center;
        color: #aab4c3;
        font-size: 0.82rem;
        margin: 17px 0 14px;
    }

    .visual-disclaimer {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 13px 16px;
        border: 1px solid #334e78;
        border-radius: 12px;
        background: rgba(25, 46, 77, 0.38);
        color: #b9d3f5;
        font-size: 0.82rem;
        line-height: 1.45;
    }

    .visual-disclaimer-icon {
        font-size: 1.1rem;
        flex: 0 0 auto;
    }

    @media (max-width: 1050px) {
        .visual-grid {
            grid-template-columns: 1fr 1fr;
        }

        .visual-card:last-child {
            grid-column: 1 / -1;
            max-width: 520px;
            width: 100%;
            justify-self: center;
        }
    }

    @media (max-width: 700px) {
        .visual-grid {
            grid-template-columns: 1fr;
        }

        .visual-card:last-child {
            grid-column: auto;
            max-width: none;
        }

        .visual-image {
            height: 290px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)




# ============================================================
# MAIN APPLICATION
# ============================================================

if active_image_bytes is not None:

    if uploaded_file is None and st.session_state.prediction_result is not None:
        st.caption("↩️ Previous screening restored — your uploaded image and result are still available.")

    # ========================================================
    # READ ORIGINAL IMAGE
    # ========================================================

    try:

        original_image = Image.open(
            io.BytesIO(active_image_bytes)
        ).convert("RGB")

    except Exception as e:

        st.error(
            f"Could not open the uploaded image: {e}"
        )

        st.stop()


    # ========================================================
    # IMAGE QUALITY VALIDATION
    # ========================================================

    if st.session_state.image_quality_result is None:
        st.session_state.image_quality_result = validate_retina_image_quality(
            original_image
        )

    quality_result = st.session_state.image_quality_result

    if not quality_result.get("valid", False):
        st.markdown(
            """
            <div class="quality-warning-card">
                <div class="quality-warning-title">⚠️ Image Quality Insufficient for Screening</div>
                <div class="quality-warning-subtitle">
                    The uploaded image did not pass the basic pre-screening quality checks.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for quality_issue in quality_result.get("issues", []):
            st.error(f"• {quality_issue}")

        st.info(
            "Please upload a clear retinal fundus photograph with the retina "
            "visible and reasonably in focus. No prediction was sent to the AI model."
        )
        st.stop()

    st.success(
        f"✅ Image quality check passed — {quality_result['width']}×{quality_result['height']} pixels. "
        "Ready for AI analysis."
    )

    # ========================================================
    # SEND IMAGE TO BACKEND
    # ========================================================

    if st.session_state.prediction_result is None:

        with st.spinner("Analyzing retinal image..."):

            try:

                response = requests.post(
                    BACKEND_URL,
                    files={
                        "file": (
                            active_image_name,
                            io.BytesIO(active_image_bytes),
                            active_image_type or "image/jpeg"
                        )
                    },
                    timeout=120
                )

                response.raise_for_status()

                result = response.json()
                st.session_state.prediction_result = result

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to the backend.\n\n"
                    "Please make sure FastAPI is running on "
                    "http://localhost:8000"
                )

                st.stop()

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ The backend took too long to respond. "
                    "Please try again."
                )

                st.stop()

            except requests.exceptions.RequestException as e:

                st.error(
                    f"❌ Backend request failed:\n\n{e}"
                )

                st.stop()

            except ValueError as e:

                st.error(
                    f"❌ Backend returned invalid JSON:\n\n{e}"
                )

                st.stop()

    else:

        result = st.session_state.prediction_result


    # ========================================================
    # GET PREDICTION DATA
    # ========================================================

    predicted_label = result.get(
        "predicted_label",
        "Unknown"
    )

    try:

        confidence = float(
            result.get(
                "confidence",
                0.0
            )
        )

    except (TypeError, ValueError):

        confidence = 0.0


    probabilities = result.get(
        "probabilities",
        {}
    )


    # ========================================================
    # SEVERITY INFORMATION
    # ========================================================

    severity_info = {

        "No DR": {
            "icon": "🟢",
            "name": "No DR",
            "status": "NO DIABETIC RETINOPATHY",
            "description": (
                "The model predicts that this retinal image "
                "is most consistent with no diabetic retinopathy."
            ),
            "recommendation": (
                "Continue routine eye screening as recommended "
                "by a qualified eye-care professional."
            )
        },

        "Mild": {
            "icon": "🟡",
            "name": "Mild DR",
            "status": "MILD SEVERITY",
            "description": (
                "The model predicts that this retinal image "
                "is most consistent with mild diabetic retinopathy."
            ),
            "recommendation": (
                "Professional eye-care evaluation is recommended."
            )
        },

        "Moderate": {
            "icon": "🟠",
            "name": "Moderate DR",
            "status": "MODERATE SEVERITY",
            "description": (
                "The model predicts that this retinal image "
                "is most consistent with moderate diabetic retinopathy."
            ),
            "recommendation": (
                "Professional eye-care evaluation is recommended."
            )
        },

        "Severe": {
            "icon": "🔴",
            "name": "Severe DR",
            "status": "HIGH SEVERITY",
            "description": (
                "The model predicts that this retinal image "
                "is most consistent with severe diabetic retinopathy."
            ),
            "recommendation": (
                "Prompt evaluation by a qualified eye-care "
                "professional is recommended."
            )
        },

        "Proliferative DR": {
            "icon": "🔴",
            "name": "Proliferative DR",
            "status": "VERY HIGH SEVERITY",
            "description": (
                "The model predicts that this retinal image "
                "is most consistent with proliferative "
                "diabetic retinopathy."
            ),
            "recommendation": (
                "Prompt evaluation by a qualified eye-care "
                "professional is strongly recommended."
            )
        }
    }


    # ========================================================
    # CURRENT RESULT
    # ========================================================

    current_result = severity_info.get(
        predicted_label,
        {
            "icon": "🔵",
            "name": predicted_label,
            "status": "MODEL PREDICTION",
            "description": (
                "The model generated a severity prediction "
                "for this retinal image."
            ),
            "recommendation": (
                "Please consult a qualified eye-care "
                "professional for further evaluation."
            )
        }
    )


    # ========================================================
    # NORMALIZE PROBABILITIES
    # ========================================================
    #
    # The backend normally returns values like:
    # 0.293, 0.051, 0.117, 0.507, 0.033
    #
    # But this also safely handles:
    # 29.3, 5.1, 11.7, 50.7, 3.3
    #
    # ========================================================

    severity_levels = [
        "No DR",
        "Mild",
        "Moderate",
        "Severe",
        "Proliferative DR"
    ]

    normalized_probabilities = {}

    for label in severity_levels:

        try:

            value = float(
                probabilities.get(
                    label,
                    0.0
                )
            )

        except (TypeError, ValueError):

            value = 0.0

        # Convert percentage values to decimal
        if value > 1.0:
            value = value / 100.0

        # Keep probability between 0 and 1
        value = min(
            max(value, 0.0),
            1.0
        )

        normalized_probabilities[label] = value


    # ========================================================
    # SAVE SCREENING TO DATABASE
    # ========================================================

    screening_key = (
        active_image_name,
        patient_name,
        patient_id,
        int(patient_age),
        gender,
        screening_notes,
        predicted_label,
        round(confidence, 6),
        tuple(
            round(normalized_probabilities[level], 6)
            for level in severity_levels
        )
    )

    if st.session_state.saved_screening_key != screening_key:

        screening_data = {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "age": int(patient_age),
            "gender": gender,
            "screening_notes": screening_notes,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": normalized_probabilities
        }

        try:

            save_response = requests.post(
                SCREENINGS_URL,
                json=screening_data,
                timeout=10
            )

            save_response.raise_for_status()

            st.session_state.saved_screening_key = screening_key

            st.success(
                "✅ Screening saved to history."
            )

        except requests.exceptions.RequestException as e:

            st.warning(
                f"⚠️ Screening result could not be saved to history: {e}"
            )


    # ========================================================
    # VISUAL EXPLANATION
    # ========================================================

    st.markdown("---")

    st.markdown(
        """
        <div class="visual-section-header">
            <div class="visual-section-title">🔬 Visual Explanation</div>
            <div class="visual-section-subtitle">
                Compare the original retinal image with the model's attention map
                and an intuitive overlay view.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    heatmap_base64 = result.get("heatmap_base64")
    heatmap_image = None

    if heatmap_base64:
        try:
            heatmap_bytes = base64.b64decode(heatmap_base64)
            heatmap_image = Image.open(
                io.BytesIO(heatmap_bytes)
            ).convert("RGB")
        except Exception as e:
            st.warning(f"Could not prepare Grad-CAM visualization: {e}")

    # Create a clean third visualization by blending the attention map
    # with the original retina. This gives a more intuitive focus view.
    overlay_image = None
    if heatmap_image is not None:
        try:
            base_for_overlay = original_image.convert("RGB").resize(
                heatmap_image.size
            )
            overlay_image = Image.blend(
                base_for_overlay,
                heatmap_image,
                alpha=0.45
            )
        except Exception:
            overlay_image = None

    original_uri = pil_to_data_uri(original_image)
    heatmap_uri = pil_to_data_uri(heatmap_image) if heatmap_image is not None else None
    overlay_uri = pil_to_data_uri(overlay_image) if overlay_image is not None else None

    visual_cards = [
        {
            "icon": "🖼️",
            "title": "Original Retina",
            "subtitle": "Uploaded retinal fundus image",
            "description": "The original retinal image submitted for screening.",
            "uri": original_uri,
        }
    ]

    if heatmap_uri:
        visual_cards.append(
            {
                "icon": "🔥",
                "title": "Grad-CAM Attention",
                "subtitle": "Model attention heatmap",
                "description": (
                    "Warmer regions indicate areas receiving stronger model attention."
                ),
                "uri": heatmap_uri,
            }
        )

    if overlay_uri:
        visual_cards.append(
            {
                "icon": "🎯",
                "title": "Attention Overlay",
                "subtitle": "Grad-CAM blended with the retina",
                "description": (
                    "Combines the retinal image and attention map for easier visual interpretation."
                ),
                "uri": overlay_uri,
            }
        )

    cards_html = "".join(
        f"""
        <div class="visual-card">
            <div class="visual-card-heading">
                <span class="visual-card-icon">{card['icon']}</span>
                <div>
                    <div class="visual-card-title">{card['title']}</div>
                    <div class="visual-card-subtitle">{card['subtitle']}</div>
                </div>
            </div>
            <div class="visual-image-frame">
                <img src="{card['uri']}" class="visual-image" alt="{card['title']}" />
                <div class="visual-hover-label">⌕ Hover to zoom</div>
            </div>
            <div class="visual-card-description">{card['description']}</div>
        </div>
        """
        for card in visual_cards
    )

    st.markdown(
        f"""
        <div class="visual-grid">
            {cards_html}
        </div>
        <div class="visual-tip">💡 Hover over an image to preview it at a larger scale.</div>
        <div class="visual-disclaimer">
            <span class="visual-disclaimer-icon">ⓘ</span>
            <span>
                Grad-CAM visualizations are provided for explainability only and
                should not be treated as standalone clinical diagnostic maps.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # SCREENING RESULT
    # ========================================================

    st.markdown("---")

    st.subheader("🎯 Screening Result")

    with st.container(border=True):

        st.markdown(
            f"## {current_result['icon']} "
            f"Prediction: {current_result['name']}"
        )

        st.markdown(
            f"### Confidence: {confidence:.1%}"
        )

        st.progress(
            min(
                max(confidence, 0.0),
                1.0
            )
        )

        st.markdown("---")

        st.markdown(
            f"### {current_result['status']}"
        )

        st.write(
            current_result["description"]
        )

        st.info(
            f"💡 {current_result['recommendation']}"
        )


    # ========================================================
    # DR SEVERITY INDICATOR
    # ========================================================

    st.subheader(
        "🩺 Diabetic Retinopathy Severity"
    )

    severity_icons = [
        "🟢",
        "🟡",
        "🟠",
        "🔴",
        "🔴"
    ]

    severity_cols = st.columns(5)

    for i, level in enumerate(severity_levels):

        with severity_cols[i]:

            if predicted_label == level:

                st.success(
                    f"{severity_icons[i]}\n\n"
                    f"**{level}**\n\n"
                    "⬆️ Predicted"
                )

            else:

                st.info(
                    f"{severity_icons[i]}\n\n"
                    f"**{level}**"
                )


    # ========================================================
    # SCREENING INTERPRETATION
    # ========================================================

    st.subheader(
        "🔎 Screening Interpretation"
    )

    with st.container(border=True):

        st.write(
            current_result["description"]
        )

        st.caption(
            "This interpretation is generated from the model "
            "prediction and is not a clinical diagnosis."
        )


    # ========================================================
    # PROBABILITY BY SEVERITY
    # ========================================================

    st.subheader(
        "📊 Probability by Severity"
    )

    probability_cols = st.columns(5)

    for i, label in enumerate(severity_levels):

        probability = normalized_probabilities[label]

        with probability_cols[i]:

            st.metric(
                label,
                f"{probability:.1%}"
            )


    # ========================================================
    # PROBABILITY CHART
    # ========================================================

    st.subheader(
        "📈 Severity Probability Distribution"
    )

    chart_data = pd.DataFrame(
        {
            "Severity": severity_levels,
            "Probability": [
                normalized_probabilities[label]
                for label in severity_levels
            ]
        }
    )

    # Set severity as the index.
    # This ensures Streamlit uses the labels as categories
    # and the probabilities as the Y-axis values.
    chart_data = chart_data.set_index(
        "Severity"
    )

    st.bar_chart(
        chart_data,
        y="Probability",
        height=400
    )


    # ========================================================
    # MODEL EXPLANATION
    # ========================================================

    st.subheader(
        "🔍 Model Explanation"
    )

    st.write(
        "The Grad-CAM visualization highlights image "
        "regions that contributed more strongly to the "
        "model's prediction. It is provided for model "
        "interpretability and should not be considered "
        "a clinical explanation."
    )


    # ========================================================
    # MODEL PERFORMANCE - V3
    # ========================================================

    st.subheader(
        "📈 Model V3 Performance"
    )

    performance_col1, performance_col2 = st.columns(2)

    with performance_col1:

        st.metric(
            "Test Accuracy",
            "84.91%"
        )

    with performance_col2:

        st.metric(
            "Quadratic Weighted Kappa",
            "0.9005"
        )

    st.caption(
        "Performance measured on the held-out test set "
        "of 550 retinal images using DR Model V3."
    )


    # ========================================================
    # PATIENT SUMMARY
    # ========================================================

    st.subheader(
        "📋 Screening Information"
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.write("**Patient Name**")

        st.write(
            patient_name
            if patient_name
            else "Not provided"
        )

    with summary_col2:

        st.write("**Patient ID**")

        st.write(
            patient_id
            if patient_id
            else "Not provided"
        )

    with summary_col3:

        st.write("**Age / Gender**")

        st.write(
            f"{patient_age} years / {gender}"
        )

    if screening_notes:

        st.write("**Screening Notes**")

        st.write(
            screening_notes
        )


    # ========================================================
    # PDF SCREENING REPORT
    # ========================================================

    st.subheader(
        "📄 Screening Report"
    )

    # Prepare the Grad-CAM image for the PDF if it was returned
    # successfully by the backend.
    pdf_heatmap_image = None

    if heatmap_base64:
        try:
            pdf_heatmap_bytes = base64.b64decode(
                heatmap_base64
            )
            pdf_heatmap_image = Image.open(
                io.BytesIO(pdf_heatmap_bytes)
            ).convert("RGB")
        except Exception:
            pdf_heatmap_image = None

    try:
        pdf_bytes = create_screening_pdf(
            patient_name=patient_name,
            patient_id=patient_id,
            patient_age=patient_age,
            gender=gender,
            screening_notes=screening_notes,
            predicted_label=predicted_label,
            current_result=current_result,
            confidence=confidence,
            normalized_probabilities=normalized_probabilities,
            original_image=original_image,
            heatmap_image=pdf_heatmap_image,
        )

        report_patient_name = (
            patient_name.strip()
            if patient_name.strip()
            else "patient"
        )

        report_filename = (
            f"DR_Screening_Report_"
            f"{_safe_filename(report_patient_name)}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

        st.download_button(
            label="📥 Download Screening Report (PDF)",
            data=pdf_bytes,
            file_name=report_filename,
            mime="application/pdf",
            use_container_width=True,
        )

        st.caption(
            "The report contains the patient information, model prediction, "
            "probabilities, retinal images, Grad-CAM visualization, and disclaimer."
        )

    except Exception as e:
        st.error(
            f"Could not generate the PDF report: {e}"
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown(
        """
        <div class="warning">

        ⚠️ <b>Important:</b> This application is an
        AI-assisted educational screening demo and is
        not a medical diagnostic tool.

        <br><br>

        Results should not be used as a substitute for
        evaluation by a qualified eye-care professional.

        </div>
        """,
        unsafe_allow_html=True
    )






    # ========================================================
    # DEVELOPER CREDIT
    # ========================================================
    developer_image_uri = get_developer_image_uri()
    developer_image_html = (
        f'<img src="{developer_image_uri}" alt="Arpit Pandey" />'
        if developer_image_uri
        else ""
    )

    st.markdown(
        f"""
        <div class="developer-footer">
            <div class="developer-footer-line"></div>
            <div class="developer-footer-content">
                <span class="developer-footer-text">Developed by <b>Arpit Pandey</b></span>
                {developer_image_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FINAL UI POLISH — STEP 6 PART 8
# ============================================================

st.markdown(
    """
    <style>

    /* Cleaner overall spacing */
    .block-container {
        max-width: 1400px;
        padding-top: 1.4rem;
        padding-bottom: 3.5rem;
    }

    /* Consistent section headings */
    h1, h2, h3 {
        letter-spacing: -0.25px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #151a25, #11151e);
        border: 1px solid #2c3445;
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 108px;
    }

    div[data-testid="stMetricLabel"] {
        color: #aeb8c9;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #364055;
        transition: all 0.15s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #6d5dfc;
        transform: translateY(-1px);
    }

    /* Inputs */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] {
        border-radius: 10px;
    }

    /* Upload area */
    section[data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #3c465a;
        border-radius: 14px;
        background: #151a24;
    }

    /* History table */
    div[data-testid="stDataFrame"] {
        border: 1px solid #2c3445;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Professional informational/warning boxes */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Footer-style disclaimer */
    .warning {
        border: 1px solid #4a4031;
        background: linear-gradient(135deg, #211e1a, #181713);
        border-radius: 14px;
    }

    /* Mobile */
    @media (max-width: 760px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        div[data-testid="stMetric"] {
            min-height: auto;
        }
    }


    /* ========================================================
       3D PROFESSIONAL UI SYSTEM
       Subtle depth, lift and glass effects across the app.
       ======================================================== */

    /* Main application surface */
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(79, 70, 229, 0.08), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(37, 99, 235, 0.07), transparent 25%),
            #0b0e14;
    }

    /* Top navigation gets a layered 3D/glass treatment */
    .top-nav {
        position: relative;
        overflow: hidden;
        transform: translateZ(0);
        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.32),
            0 2px 0 rgba(255, 255, 255, 0.04) inset,
            0 -1px 0 rgba(0, 0, 0, 0.35) inset;
        transition: transform 0.25s ease, box-shadow 0.25s ease,
                    border-color 0.25s ease;
    }

    .top-nav::before {
        content: "";
        position: absolute;
        top: 0;
        left: 5%;
        right: 5%;
        height: 1px;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.24),
            transparent
        );
    }

    .top-nav:hover {
        transform: translateY(-3px);
        border-color: #3d4b66;
        box-shadow:
            0 18px 40px rgba(0, 0, 0, 0.40),
            0 2px 0 rgba(255, 255, 255, 0.05) inset;
    }

    .top-nav-logo {
        box-shadow:
            0 8px 18px rgba(59, 130, 246, 0.28),
            0 1px 0 rgba(255, 255, 255, 0.25) inset;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .top-nav:hover .top-nav-logo {
        transform: translateY(-2px) rotate(-2deg);
        box-shadow:
            0 12px 24px rgba(59, 130, 246, 0.34),
            0 1px 0 rgba(255, 255, 255, 0.28) inset;
    }

    /* Navigation selector */
    div[data-testid="stRadio"] > div {
        box-shadow:
            0 8px 20px rgba(0, 0, 0, 0.24),
            0 1px 0 rgba(255, 255, 255, 0.035) inset,
            0 -1px 0 rgba(0, 0, 0, 0.30) inset;
    }

    div[data-testid="stRadio"] label {
        transition: transform 0.20s ease, background 0.20s ease,
                    box-shadow 0.20s ease, color 0.20s ease;
    }

    div[data-testid="stRadio"] label:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 12px rgba(0, 0, 0, 0.22);
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        position: relative;
        background: linear-gradient(145deg, #171c29, #10141d);
        border: 1px solid #2d3748;
        border-radius: 16px;
        padding: 15px 18px;
        min-height: 105px;
        box-sizing: border-box;
        box-shadow:
            0 10px 24px rgba(0, 0, 0, 0.25),
            0 2px 0 rgba(255, 255, 255, 0.035) inset,
            0 -2px 0 rgba(0, 0, 0, 0.25) inset;
        transition: transform 0.22s ease, box-shadow 0.22s ease,
                    border-color 0.22s ease;
    }

    div[data-testid="metric-container"]::after {
        content: "";
        position: absolute;
        left: 14px;
        right: 14px;
        bottom: 7px;
        height: 1px;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,0.08),
            transparent
        );
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: #465575;
        box-shadow:
            0 17px 34px rgba(0, 0, 0, 0.34),
            0 2px 0 rgba(255, 255, 255, 0.045) inset;
    }

    /* Buttons throughout the application */
    div.stButton > button {
        position: relative;
        overflow: hidden;
        border: 1px solid #39455c;
        border-radius: 11px;
        background: linear-gradient(145deg, #171c27, #11151e);
        color: #f8fafc;
        box-shadow:
            0 7px 16px rgba(0, 0, 0, 0.25),
            0 1px 0 rgba(255, 255, 255, 0.05) inset,
            0 -2px 0 rgba(0, 0, 0, 0.25) inset;
        transition: transform 0.18s ease, box-shadow 0.18s ease,
                    border-color 0.18s ease, background 0.18s ease;
    }

    div.stButton > button::before {
        content: "";
        position: absolute;
        top: 0;
        left: -120%;
        width: 70%;
        height: 100%;
        background: linear-gradient(
            100deg,
            transparent,
            rgba(255,255,255,0.08),
            transparent
        );
        transform: skewX(-18deg);
        transition: left 0.45s ease;
        pointer-events: none;
    }

    div.stButton > button:hover {
        transform: translateY(-3px);
        border-color: #5b6d93;
        background: linear-gradient(145deg, #1b2230, #121824);
        box-shadow:
            0 12px 25px rgba(0, 0, 0, 0.34),
            0 1px 0 rgba(255, 255, 255, 0.06) inset;
    }

    div.stButton > button:hover::before {
        left: 140%;
    }

    div.stButton > button:active {
        transform: translateY(1px) scale(0.99);
        box-shadow: 0 4px 9px rgba(0,0,0,0.28);
    }

    /* Text inputs, number input, selectbox and textarea */
    div[data-baseweb="input"],
    div[data-baseweb="select"],
    div[data-baseweb="textarea"] {
        border-radius: 11px;
        transition: transform 0.18s ease, box-shadow 0.18s ease,
                    border-color 0.18s ease;
    }

    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"]:focus-within,
    div[data-baseweb="textarea"]:focus-within {
        transform: translateY(-2px);
        box-shadow:
            0 8px 18px rgba(0, 0, 0, 0.24),
            0 0 0 1px rgba(99, 102, 241, 0.28);
    }

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {
        border: 1px solid #354057;
        border-radius: 14px;
        background: linear-gradient(145deg, #181d29, #11151e);
        box-shadow:
            0 10px 22px rgba(0, 0, 0, 0.25),
            0 1px 0 rgba(255,255,255,0.04) inset;
        transition: transform 0.22s ease, border-color 0.22s ease,
                    box-shadow 0.22s ease;
    }

    section[data-testid="stFileUploaderDropzone"]:hover {
        transform: translateY(-3px);
        border-color: #536487;
        box-shadow: 0 15px 30px rgba(0,0,0,0.32);
    }

    /* Alerts / messages */
    div[data-testid="stAlert"] {
        border-radius: 13px;
        box-shadow:
            0 8px 20px rgba(0, 0, 0, 0.22),
            0 1px 0 rgba(255,255,255,0.035) inset;
    }

    /* Progress bars */
    div[data-testid="stProgress"] > div {
        border-radius: 999px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.30);
    }

    div[data-testid="stProgress"] div[role="progressbar"] {
        border-radius: 999px;
        box-shadow: 0 2px 9px rgba(59, 130, 246, 0.22);
    }

    /* Dataframe / history table */
    div[data-testid="stDataFrame"] {
        border: 1px solid #2d3748;
        border-radius: 14px;
        overflow: hidden;
        box-shadow:
            0 10px 24px rgba(0, 0, 0, 0.25),
            0 1px 0 rgba(255,255,255,0.035) inset;
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }

    div[data-testid="stDataFrame"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.30);
    }

    /* Images displayed through Streamlit outside the custom visual cards */
    div[data-testid="stImage"] img {
        border-radius: 13px;
        border: 1px solid rgba(255,255,255,0.20);
        box-shadow:
            0 10px 24px rgba(0,0,0,0.30),
            0 1px 0 rgba(255,255,255,0.06) inset;
        transition: transform 0.28s ease, filter 0.28s ease,
                    box-shadow 0.28s ease;
    }

    div[data-testid="stImage"] img:hover {
        transform: translateY(-4px) scale(1.018);
        filter: brightness(1.035) saturate(1.025);
        box-shadow:
            0 17px 34px rgba(0,0,0,0.38),
            0 0 0 1px rgba(255,255,255,0.14);
    }

    /* Section headings get subtle depth without excessive animation */
    .history-page-title,
    .main-title,
    .visual-section-title {
        text-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
    }

    /* Custom visual cards: slightly deeper 3D treatment */
    .visual-card {
        box-shadow:
            0 12px 28px rgba(0, 0, 0, 0.30),
            0 2px 0 rgba(255,255,255,0.035) inset,
            0 -2px 0 rgba(0,0,0,0.25) inset;
        transform: translateZ(0);
    }

    .visual-card:hover {
        transform: translateY(-7px) perspective(900px) rotateX(1deg);
        box-shadow:
            0 20px 40px rgba(0, 0, 0, 0.40),
            0 2px 0 rgba(255,255,255,0.05) inset;
    }

    .visual-image-frame {
        box-shadow:
            0 0 0 1px rgba(255,255,255,0.10),
            0 10px 22px rgba(0,0,0,0.34),
            0 2px 0 rgba(255,255,255,0.10) inset;
        transition: transform 0.28s ease, box-shadow 0.28s ease;
    }

    .visual-image-frame:hover {
        transform: translateY(-2px) perspective(900px) rotateX(1deg);
        box-shadow:
            0 0 0 1px rgba(255,255,255,0.18),
            0 16px 30px rgba(0,0,0,0.42),
            0 2px 0 rgba(255,255,255,0.12) inset;
    }

    /* Respect users who prefer reduced motion */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)
