# app.py
import streamlit as st
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)
from langdetect import detect
from gtts import gTTS
import tempfile, os, re
from pydub import AudioSegment
import speech_recognition as sr

# ────────────────────────────────────────────────────────────────
# Page setup
st.set_page_config(page_title="TextLearn Studio", layout="wide")
st.title("🌍 InsightQ Engine")

# ────────────────────────────────────────────────────────────────
# Sidebar
st.sidebar.header("Options")
use_speech = st.sidebar.checkbox("🎤 Use microphone", value=False)
question_types = st.sidebar.multiselect(
    "Select question types to keep:",
    ["Who", "When", "Where", "Why", "How", "Which"," Whose"," Whom"," Can", " Could", " Will", " Would", " Is", " Are", " Do", " Does", " Did", " How much", " How many", " How long", " How often", " What kind", " Other"],
    default=["Who",  "When", "Where", "Why", "How"],
)
tts_engine = st.sidebar.selectbox("Text-to-Speech engine", ["gTTS", "pyttsx3"], index=0)

# ────────────────────────────────────────────────────────────────
# Cached models
@st.cache_resource(show_spinner=False)
def load_translation_model():
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_qg_model():
    # Better model for Any-type question generation
    qg_model_name = "iarfmoose/t5-base-question-generator"
    qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_name)
    qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model_name)
    return pipeline(
        "text2text-generation",
        model=qg_model,
        tokenizer=qg_tokenizer,
        do_sample=True, 
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        device=-1,
    )

# ────────────────────────────────────────────────────────────────
# Input
if use_speech:
    st.write("Record or upload audio (wav/mp3) containing a paragraph.")
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "ogg"])
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_file.getvalue())
            temp_audio_path = f.name

        r = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio = r.record(source)
        try:
            text = r.recognize_google(audio)
            st.markdown("**Recognized text:**")
            st.write(text)
        except Exception as e:
            st.error(f"Speech-to-text error: {e}")
            st.stop()
    else:
        st.info("Upload an audio file to continue.")
        st.stop()
else:
    text = st.text_area("✍️ Type or paste a paragraph in ANY language:", height=200)

if not text or text.strip() == "":
    st.info("Please enter text or upload audio above.")
    st.stop()

# ────────────────────────────────────────────────────────────────
# 1️⃣ Language detection
try:
    lang = detect(text)
except Exception:
    lang = "unknown"
st.write(f"**Detected language:** `{lang}`")

# ────────────────────────────────────────────────────────────────
# 2️⃣ Translation → English
translation_model, translation_tokenizer = load_translation_model()
translated_text = text

if lang not in ["en", "unknown"]:
    lang_map_normalize = {
        "zh-cn": "zh",
        "zh-tw": "zh",
        "pt-br": "pt",
        "he": "he",
    }
    src_lang = lang_map_normalize.get(lang, lang)
    tgt_lang = "en"

    try:
        if hasattr(translation_tokenizer, "get_lang_id"):
            forced_bos_id = translation_tokenizer.get_lang_id(tgt_lang)
        else:
            forced_bos_id = translation_tokenizer.lang_code_to_id.get(
                tgt_lang, translation_tokenizer.lang_code_to_id.get(tgt_lang.lower())
            )

        try:
            translation_tokenizer.src_lang = src_lang
        except Exception:
            pass

        encoded = translation_tokenizer(text, return_tensors="pt", padding=True)
        generated_tokens = translation_model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_id,
            max_length=512,
            num_beams=4,
            early_stopping=True,
        )
        translated_text = translation_tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        )
    except Exception as e:
        st.warning(f"Translation failed: {e}. Using original text.")
        translated_text = text

st.markdown("### English Text (translated if needed):")
st.write(translated_text)

# ──────────────────────────────────
# 3️⃣ Question Generation
qg_pipeline = load_qg_model()
num_qs = st.sidebar.slider("Number of questions to generate", 1, 10, 5)

# Use proper "context:" prefix for cleaner question generation
prompt = f"Generate a diverse set of questions of different types
(yes/no, explanation, opinion, reasoning,and WH) based on the following text:\n\n{translated_text}"

with st.spinner("Generating questions..."):
    out = qg_pipeline(
        prompt,
        max_length=256,
        num_return_sequences=num_qs,
        num_beams=num_qs,
        clean_up_tokenization_spaces=True,
    )

raw_qs = [o["generated_text"].strip() for o in out]

# ────────────────────────────────────────────────────────────────
# Clean and filter questions
cleaned_qs = []
for q in raw_qs:
    q = re.sub(r"^(question\s*[:\-]?\s*)", "", q, flags=re.I).strip()
    # Keep only up to the first '?'
    if "?" in q:
        q = q.split("?")[0].strip() + "?"
    cleaned_qs.append(q)

questions = [q for q in cleaned_qs if q.strip()]
def filter_by_type(q):
    qlow = q.strip().lower()
    if not qlow.endswith("?"):
        qlow += "?"
    if not question_types:
        return True
    return any(qlow.startswith(qt.lower()) for qt in question_types)

filtered = [q for q in questions if filter_by_type(q)]

if not filtered:
    st.warning("No questions matched the selected types. Showing all generated questions.")
    filtered = questions

st.markdown("### 🧠 Generated Questions")
for i, q in enumerate(filtered, start=1):
    st.write(f"{i}. {q}")

# ────────────────────────────────────────────────────────────────
# 4️⃣ Text-to-Speech
if st.button("🔊 Play Questions (TTS)"):
    combined = " ".join(filtered)
    try:
        if tts_engine == "gTTS":
            tts = gTTS(combined, lang="en")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp.name)
            st.audio(tmp.name)
        else:
            import pyttsx3
            engine = pyttsx3.init()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            engine.save_to_file(combined, tmp.name)
            engine.runAndWait()
            st.audio(tmp.name)
    except Exception as e:
        st.error(f"TTS error: {e}")

st.success("✅ Done")
