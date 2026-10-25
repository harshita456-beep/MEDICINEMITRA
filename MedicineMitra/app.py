
import os
import re
import io
import json
import time
import base64
import zipfile
import calendar
from datetime import date
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="MedicineMitra",
    page_icon="💊",
    layout="wide",
)

API_KEY = os.getenv("SARVAM_API_KEY", "").strip()

LANGUAGES = {
    "Hindi": {
        "code": "hi-IN",
        "translate": "hi-IN",
        "voice": "hi-IN",
    },
    "Bengali": {
        "code": "bn-IN",
        "translate": "bn-IN",
        "voice": "bn-IN",
    },
    "Tamil": {
        "code": "ta-IN",
        "translate": "ta-IN",
        "voice": "ta-IN",
    },
    "Telugu": {
        "code": "te-IN",
        "translate": "te-IN",
        "voice": "te-IN",
    },
    "Marathi": {
        "code": "mr-IN",
        "translate": "mr-IN",
        "voice": "mr-IN",
    },
    "Kannada": {
        "code": "kn-IN",
        "translate": "kn-IN",
        "voice": "kn-IN",
    },
    "Malayalam": {
        "code": "ml-IN",
        "translate": "ml-IN",
        "voice": "ml-IN",
    },
    "Gujarati": {
        "code": "gu-IN",
        "translate": "gu-IN",
        "voice": "gu-IN",
    },
    "Punjabi": {
        "code": "pa-IN",
        "translate": "pa-IN",
        "voice": "pa-IN",
    },
    "English": {
        "code": "en-IN",
        "translate": "en-IN",
        "voice": "en-IN",
    },
}

if not API_KEY:
    st.error(
        "SARVAM_API_KEY is missing. Add it to your .env file "
        "and restart Streamlit."
    )
    st.stop()

client = SarvamAI(api_subscription_key=API_KEY)


# ---------------- GENERAL HELPERS ----------------

def value_of(obj, *names):
    """Read a value from an SDK response object or dictionary."""
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def response_text(response):
    """Get text from a Sarvam chat response."""
    choices = value_of(response, "choices")
    if choices:
        message = value_of(choices[0], "message")
        content = value_of(message, "content")
        if isinstance(content, str):
            return content
    raise ValueError("Sarvam returned no readable chat response.")


def extract_json(text):
    """Parse JSON even if the model wraps it in Markdown fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise ValueError("The AI response did not contain valid JSON.")


# ---------------- SARVAM OCR ----------------

def extract_ocr_text(uploaded_file):
    """
    Submit the uploaded image/PDF to Sarvam Document AI.
    Poll for completion and read the resulting text files.
    """
    filename = uploaded_file.name
    content = uploaded_file.getvalue()
    mime_type = uploaded_file.type or "application/octet-stream"

    if not content:
        raise ValueError("The uploaded file is empty.")

    # Document AI returns a job; its status must be checked.
    job = client.doc_ai.digitise(
        file=[(filename, content, mime_type)],
        language="hi-IN",
        output_format="md",
    )

    job_id = value_of(job, "job_id", "id")
    if not job_id:
        raise RuntimeError(
            "Sarvam did not return a document job ID. "
            "Check the Document AI API response and your account access."
        )

    progress = st.progress(0, text="Sarvam is reading the medicine label…")
    status_text = st.empty()

    status_response = None
    terminal_statuses = {
        "completed",
        "partially_completed",
        "failed",
        "rejected",
    }

    for attempt in range(60):
        status_response = client.doc_ai.get_status(job_id=job_id)
        status = value_of(status_response, "status", "job_status")
        status = str(status or "").lower()

        progress.progress(
            min(95, int((attempt + 1) / 60 * 95)),
            text="Sarvam is reading the medicine label…",
        )
        status_text.caption(f"OCR status: {status or 'processing'}")

        if status in terminal_statuses:
            break

        time.sleep(2)

    else:
        progress.empty()
        status_text.empty()
        raise TimeoutError(
            "OCR is taking too long. Try again with a clearer, smaller image."
        )

    progress.empty()
    status_text.empty()

    if status in {"failed", "rejected"}:
        detail = value_of(
            status_response, "message", "error", "failure_reason"
        )
        raise RuntimeError(f"Sarvam OCR failed: {detail or status}")

    # Ask Sarvam for the generated output download URL.
    download_info = client.doc_ai.get_download_url(job_id=job_id)
    download_url = value_of(
        download_info, "download_url", "url", "signed_url"
    )

    if not download_url:
        raise RuntimeError(
            "Sarvam completed the OCR job but did not return a download URL. "
            "Check the Document AI response fields for your SDK version."
        )

    response = requests.get(download_url, timeout=60)
    response.raise_for_status()
    downloaded = response.content

    # Document AI output may be delivered as a ZIP archive.
    if zipfile.is_zipfile(io.BytesIO(downloaded)):
        pieces = []
        with zipfile.ZipFile(io.BytesIO(downloaded)) as archive:
            for name in archive.namelist():
                if name.endswith("/"):
                    continue

                lower_name = name.lower()
                if lower_name.endswith(
                    (".md", ".txt", ".json", ".html")
                ):
                    raw = archive.read(name)
                    text = raw.decode("utf-8", errors="replace")

                    if lower_name.endswith(".json"):
                        try:
                            data = json.loads(text)
                            text = json.dumps(
                                data, ensure_ascii=False, indent=2
                            )
                        except json.JSONDecodeError:
                            pass

                    pieces.append(f"\n--- {name} ---\n{text}")

        ocr_text = "\n".join(pieces).strip()
    else:
        # Some API versions may return a text file directly.
        ocr_text = downloaded.decode("utf-8", errors="replace").strip()

    if not ocr_text:
        raise ValueError(
            "Sarvam returned no readable text. Try a clearer image "
            "showing the full medicine name and expiry date."
        )

    return ocr_text


# ---------------- SARVAM-105B PARSING ----------------

FIELDS = [
    "medicine_name",
    "active_ingredient",
    "strength",
    "dosage_form",
    "manufacturer",
    "batch_number",
    "manufacturing_date",
    "expiry_date",
    "mrp",
    "pack_size",
    "storage_instructions",
    "prescription_warning",
    "printed_indications",
    "likely_common_uses",
    "use_explanation",
    "side_effects_on_label",
    "warnings_on_label",
    "confidence",
    "uncertain_fields",
    "safety_note",
]


def parse_medicine_with_sarvam(ocr_text):
    prompt = f"""
You are an Indian medicine-package text interpreter.
Use ONLY the OCR text below as evidence for package-specific fields.

Return exactly one valid JSON object with these keys:
{json.dumps(FIELDS)}

Rules:
- Never invent a brand, ingredient, strength, manufacturer, date, price,
  batch number, indication, side effect, or warning.
- Use "Not found" for unavailable scalar fields.
- Use [] for uncertain_fields if there are none; otherwise list field names.
- Separate "printed_indications" (claims actually printed on the package)
  from "likely_common_uses" (general interpretation of a clearly identified
  active ingredient and dosage form).
- Do not state that a use is confirmed by the package unless it is printed.
- If the medicine or active ingredient cannot be identified confidently,
  set likely_common_uses to "Unable to reliably determine from this label".
- If the ingredient is clear, describe common uses cautiously and briefly.
  Do not diagnose the user or advise whether they personally should take it.
- Never recommend a dose, dose schedule, or treatment duration.
- Do not infer a medicine's identity from a partially visible brand name alone.
- Preserve Indian label wording, strengths, and rupee prices.
- Do not calculate or guess an expiry date from unreadable characters.
- "confidence" must be "High", "Medium", or "Low", based on the OCR evidence.
- Keep "use_explanation" understandable to a non-medical reader.
- "safety_note" must remind the reader that the interpretation is AI-generated
  and should be checked with a pharmacist or doctor.
- All JSON values must be strings or arrays of strings, except where natural.

OCR TEXT:
<ocr>
{ocr_text[:16000]}
</ocr>
"""

    response = client.chat.completions(
        model="sarvam-105b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract and interpret medicine package text cautiously. "
                    "Do not fabricate missing facts. Output valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1800,
    )

    data = extract_json(response_text(response))

    for field in FIELDS:
        if field not in data or data[field] is None:
            data[field] = [] if field == "uncertain_fields" else "Not found"

    data["information_status"] = (
        "AI interpretation of package text; not independently verified."
    )
    return data


# ---------------- EXPIRY CHECK ----------------

def check_expiry(expiry_text):
    """
    Deterministically checks only unambiguous month/year or date formats.
    Returns unknown rather than guessing when the date is unclear.
    """
    if not isinstance(expiry_text, str):
        return "Unable to determine expiry from the extracted text."

    text = expiry_text.strip().upper()
    if not text or text in {"NOT FOUND", "UNKNOWN", "UNCLEAR"}:
        return "Expiry date not found or unreadable."

    today = date.today()

    # Examples: 08/2027, 08-2027, 08.2027
    match = re.search(r"(?<!\d)(0?[1-9]|1[0-2])\s*[/.-]\s*(20\d{2})(?!\d)", text)
    if match:
        month = int(match.group(1))
        year = int(match.group(2))
        last_day = calendar.monthrange(year, month)[1]
        expiry_date = date(year, month, last_day)

        if today > expiry_date:
            return f"EXPIRED — label indicates {month:02d}/{year}."
        return (
            f"Not past the printed expiry month ({month:02d}/{year}). "
            "Check the physical package and pharmacist's advice."
        )

    # Examples: AUG 2027, AUGUST 2027
    month_names = {
        name.upper(): number
        for number, name in enumerate(calendar.month_name)
        if name
    }
    month_names.update({
        name.upper(): number
        for number, name in enumerate(calendar.month_abbr)
        if name
    })

    for name, month in month_names.items():
        match = re.search(rf"\b{name}\b[\s,/-]*(20\d{{2}})\b", text)
        if match:
            year = int(match.group(1))
            expiry_date = date(
                year, month, calendar.monthrange(year, month)[1]
            )
            if today > expiry_date:
                return f"EXPIRED — label indicates {name} {year}."
            return (
                f"Not past the printed expiry month ({name} {year}). "
                "Check the physical package and pharmacist's advice."
            )

    # Do not guess formats such as 03/04/2027, which may be ambiguous.
    return (
        "Expiry date could not be safely interpreted. "
        "Check the original package with a pharmacist."
    )


# ---------------- SARVAM TRANSLATION ----------------

def translate_text(text, target_language):
    if not text.strip():
        return ""

    if target_language == "en-IN":
        return text

    result = client.text.translate(
        input=text[:4500],
        source_language_code="en-IN",
        target_language_code=target_language,
        model="sarvam-translate:v1",
    )

    translated = value_of(result, "translated_text")
    if not translated:
        raise RuntimeError("Sarvam Translate returned no translated text.")
    return translated


# ---------------- SARVAM BULBUL TTS ----------------

def text_to_speech(text, language_code):
    if not text.strip():
        raise ValueError("There is no text to read aloud.")

    result = client.text_to_speech.convert(
        text=text[:2000],
        language_code=language_code,
        model="bulbul:v3",
    )

    audios = value_of(result, "audios")
    if not audios:
        raise RuntimeError("Sarvam TTS returned no audio.")
    return base64.b64decode(audios[0])


# ---------------- SUMMARY ----------------

def make_summary(medicine, expiry_status):
    def field(name):
        value = medicine.get(name, "Not found")
        if isinstance(value, list):
            return ", ".join(map(str, value)) if value else "Not found"
        return str(value)

    return f"""
Medicine name: {field("medicine_name")}
Active ingredient: {field("active_ingredient")}
Strength: {field("strength")}
Form: {field("dosage_form")}
Common uses (AI interpretation, not verified): {field("likely_common_uses")}
Explanation: {field("use_explanation")}
Indications printed on package: {field("printed_indications")}
Manufacturer: {field("manufacturer")}
Batch number: {field("batch_number")}
Manufacturing date: {field("manufacturing_date")}
Expiry printed on package: {field("expiry_date")}
Expiry check: {expiry_status}
Maximum retail price: {field("mrp")}
Pack size: {field("pack_size")}
Storage: {field("storage_instructions")}
Prescription warning: {field("prescription_warning")}
Side effects printed on package: {field("side_effects_on_label")}
Warnings printed on package: {field("warnings_on_label")}

Safety: This is an AI interpretation of the package, not verified medical
advice. Do not use this explanation to choose a medicine or dose. Confirm
the medicine and its use with a pharmacist or doctor.
""".strip()


# ---------------- STREAMLIT UI ----------------

st.title("💊 MedicineMitra")
st.subheader("Understand your Indian medicine label in your language")
st.write(
    "Upload a clear photo of a medicine strip, bottle, or box. "
    "MedicineMitra reads the label, explains the extracted information, "
    "translates it, and can read it aloud."
)

st.warning(
    "For information only. Do not use this app to decide a dose, "
    "start or stop a medicine, or determine whether a medicine is safe for you."
)

language_name = st.selectbox(
    "Choose the language for translation and voice",
    list(LANGUAGES.keys()),
    index=0,
)
target_language = LANGUAGES[language_name]["translate"]
voice_language = LANGUAGES[language_name]["voice"]

uploaded_file = st.file_uploader(
    "Upload medicine packaging",
    type=["png", "jpg", "jpeg", "webp", "pdf"],
    help="Use a sharp, well-lit image. Include the ingredient, strength, and expiry.",
)

if uploaded_file:
    if uploaded_file.type and uploaded_file.type.startswith("image/"):
        st.image(
            uploaded_file.getvalue(),
            caption="Uploaded medicine package",
            use_container_width=True,
        )
    elif uploaded_file.type == "application/pdf":
        st.info("PDF uploaded. OCR will process the document.")

    if st.button("🔍 Read and explain medicine", type="primary"):
        try:
            with st.spinner("Step 1/3: Reading package text with Sarvam…"):
                ocr_text = extract_ocr_text(uploaded_file)

            st.session_state["ocr_text"] = ocr_text

            with st.spinner(
                "Step 2/3: Sarvam-105B is extracting label details "
                "and preparing a cautious explanation…"
            ):
                medicine = parse_medicine_with_sarvam(ocr_text)

            expiry_status = check_expiry(medicine.get("expiry_date", ""))
            summary = make_summary(medicine, expiry_status)

            st.session_state["medicine"] = medicine
            st.session_state["expiry_status"] = expiry_status
            st.session_state["summary"] = summary

        except Exception as error:
            st.error(
                "Could not process this package. Check your API key, "
                "Sarvam account access, internet connection, and uploaded file."
            )
            st.code(str(error))

if "ocr_text" in st.session_state:
    with st.expander("View raw OCR text"):
        st.text(st.session_state["ocr_text"])

if "medicine" in st.session_state:
    medicine = st.session_state["medicine"]
    expiry_status = st.session_state["expiry_status"]

    st.divider()
    st.header("Medicine label results")

    st.caption(
        "Extracted and interpreted by Sarvam AI. Verify the physical package."
    )

    if expiry_status.startswith("EXPIRED"):
        st.error(expiry_status)
    elif "could not" in expiry_status.lower() or "not found" in expiry_status.lower():
        st.warning(expiry_status)
    else:
        st.info(expiry_status)

    if medicine.get("confidence") == "Low":
        st.warning(
            "The model reports low confidence. Retake the photo and include "
            "the full label before relying on these extracted fields."
        )

    col1, col2 = st.columns(2)

    display_fields = [
        ("Medicine name", "medicine_name"),
        ("Active ingredient", "active_ingredient"),
        ("Strength", "strength"),
        ("Dosage form", "dosage_form"),
        ("Manufacturer", "manufacturer"),
        ("Batch number", "batch_number"),
        ("Manufacturing date", "manufacturing_date"),
        ("Expiry date on label", "expiry_date"),
        ("MRP", "mrp"),
        ("Pack size", "pack_size"),
        ("Storage instructions", "storage_instructions"),
        ("Prescription warning", "prescription_warning"),
    ]

    for index, (label, key) in enumerate(display_fields):
        value = medicine.get(key, "Not found")
        if isinstance(value, list):
            value = ", ".join(map(str, value)) if value else "Not found"

        with (col1 if index % 2 == 0 else col2):
            st.markdown(f"**{label}**")
            st.write(value or "Not found")

    st.subheader("What is it commonly used for?")
    st.write(medicine.get("use_explanation", "Not found"))

    st.markdown("**AI-generated common-use interpretation**")
    st.write(medicine.get("likely_common_uses", "Not found"))

    st.markdown("**Indications actually printed on the package**")
    st.write(medicine.get("printed_indications", "Not found"))

    st.markdown("**Side effects printed on the package**")
    st.write(medicine.get("side_effects_on_label", "Not found"))

    st.markdown("**Warnings printed on the package**")
    st.write(medicine.get("warnings_on_label", "Not found"))

    st.markdown("**Fields requiring manual verification**")
    uncertain = medicine.get("uncertain_fields", [])
    if uncertain:
        st.write(", ".join(uncertain))
    else:
        st.write("No uncertain fields were reported by the model.")

    st.caption(
        "AI confidence: "
        + str(medicine.get("confidence", "Not found"))
        + " — this is not a calibrated medical confidence score."
    )

    st.divider()
    st.header(f"Translated summary — {language_name}")

    if st.button("🌐 Translate summary"):
        try:
            with st.spinner("Translating with Sarvam Translate…"):
                translated = translate_text(
                    st.session_state["summary"], target_language
                )
            st.session_state["translated_summary"] = translated
        except Exception as error:
            st.error(f"Translation failed: {error}")

    if "translated_summary" in st.session_state:
        st.write(st.session_state["translated_summary"])

        if st.button("🔊 Read summary aloud"):
            try:
                with st.spinner("Generating speech with Sarvam Bulbul…"):
                    audio = text_to_speech(
                        st.session_state["translated_summary"],
                        voice_language,
                    )
                st.audio(audio, format="audio/wav")
            except Exception as error:
                st.error(
                    "Speech generation failed. Check Sarvam TTS model access "
                    "and language support."
                )
                st.code(str(error))

    with st.expander("View structured JSON returned by Sarvam-105B"):
        st.json(medicine)

    st.download_button(
        "Download extracted medicine details (JSON)",
        data=json.dumps(medicine, ensure_ascii=False, indent=2),
        file_name="medicinemitra_results.json",
        mime="application/json",
    )

st.divider()
st.caption(
    "MedicineMitra is an educational label-reading aid. It cannot verify "
    "authenticity, detect counterfeit medicines, confirm that a medicine is "
    "appropriate for you, or replace advice from a licensed pharmacist or doctor."
)
