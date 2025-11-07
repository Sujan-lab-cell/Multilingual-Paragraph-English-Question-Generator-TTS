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
import speech_recognition as sr
import math

# Optional: use torch to detect GPU (pipeline uses device arg)
try:
    import torch
    _has_cuda = torch.cuda.is_available()
except Exception:
    _has_cuda = False

# ────────────────────────────────────────────────────────────────
# Page setup
st.set_page_config(page_title="TextLearn Studio", layout="wide")
st.title("🌍 InsightQ Engine")

# ────────────────────────────────────────────────────────────────
# Sidebar controls
st.sidebar.header("Options")
use_speech = st.sidebar.checkbox("🎤 Use microphone", value=False)
question_types = st.sidebar.multiselect(
    "Select question types to keep:",
    ["Who", "When", "Where", "Why","what", "How", "Which", "Whose", "Whom", "Can", "Could", "Will", "Would", "Is", "Are", "Do", "Does", "Did", "How much", "How many", "How long", "How often", "What kind", "Other"],
    default=["Who", "When", "Where", "Why", "How"],
)
tts_engine = st.sidebar.selectbox("Text-to-Speech engine", ["gTTS", "pyttsx3"], index=0)

st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox(
    "Choose Question Generation Model:",
    ["T5-Base (iarfmoose/t5-base-question-generator)", "LongT5 (google/long-t5-tglobal-base)", "LED (allenai/led-base-16384)"],
    index=0,
)

st.sidebar.markdown("### Long-text handling")
auto_chunk = st.sidebar.checkbox("Auto-chunk long text", value=True)
chunk_words = st.sidebar.slider("Chunk size (words per chunk)", 150, 500, 250)
chunk_by_tokens = st.sidebar.checkbox("Use token-aware chunking (safer)", value=True)
num_qs_per_chunk = st.sidebar.slider("Number of questions per chunk / generation call", 1, 5, 3)

# ────────────────────────────────────────────────────────────────
# Cached models
@st.cache_resource(show_spinner=False)
def load_translation_model():
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

@st.cache_resource(show_spinner=False)
def load_qg_model(choice_label):
    """
    Load the QG model & tokenizer depending on user's choice.
    Returns (pipeline_obj, tokenizer, model_name)
    """
    if "T5-Base" in choice_label:
        qg_model_name = "iarfmoose/t5-base-question-generator"
    elif "LongT5" in choice_label:
        qg_model_name = "google/long-t5-tglobal-base"
    else:
        qg_model_name = "allenai/led-base-16384"

    qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_name, use_fast=True)
    qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model_name)

    device = 0 if _has_cuda else -1
    qg_pipeline = pipeline(
        "text2text-generation",
        model=qg_model,
        tokenizer=qg_tokenizer,
        device=device,
        # generation defaults will be overridden at call time
    )
    return qg_pipeline, qg_tokenizer, qg_model_name

# ────────────────────────────────────────────────────────────────
# Input handling
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
    text = st.text_area("✍️ Type or paste a paragraph or long text in ANY language:", height=300)

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
# 2️⃣ Translation → English (if needed)
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

# ────────────────────────────────────────────────────────────────
# Helper functions: chunking (word-based and token-aware), normalization, generation
def chunk_text_by_words(text, max_words=250):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def chunk_text_by_tokens(tokenizer, text, max_input_tokens):
    """
    Token-aware chunking: splits text into chunks where tokenizer.encode length <= max_input_tokens.
    Uses a sliding window with overlap to preserve context.
    """
    # choose a small overlap (in tokens) to preserve context between chunks
    overlap_tokens = min(64, max_input_tokens // 10)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    total = len(tokens)
    while start < total:
        end = min(start + max_input_tokens, total)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        if end == total:
            break
        start = end - overlap_tokens  # overlap
    return chunks

def normalize_q(q):
    q = q.strip()
    q = re.sub(r"^(question\s*[:\-]?\s*)", "", q, flags=re.I).strip()
    if "?" in q:
        q = q.split("?")[0].strip() + "?"
    q = re.sub(r'\s+', ' ', q)
    return q

def generate_questions_from_text(qg_pipeline, qg_tokenizer, text, model_name, auto_chunk=True, chunk_words=250, token_chunk=True, per_chunk=3):
    """
    Returns deduplicated list of questions for given text.
    - If using long models that can accept entire text (LongT5/LED) and auto_chunk=False, it will attempt single-shot generation.
    - Otherwise it chunks (token-aware if requested) and generates per chunk.
    """
    is_long_model = ("long-t5" in model_name.lower()) or ("led" in model_name.lower())

    # Decide chunking strategy
    if not auto_chunk and is_long_model:
        chunks = [text]
    else:
        if token_chunk and qg_tokenizer is not None:
            # token-aware: determine safe token budget
            try:
                model_max = qg_tokenizer.model_max_length
                # leave headroom for decoder & prompt tokens; use ~80% of model_max as input budget
                safe_input = max(256, int(model_max * 0.8))
            except Exception:
                safe_input = 1024
            chunks = chunk_text_by_tokens(qg_tokenizer, text, safe_input)
        else:
            chunks = chunk_text_by_words(text, max_words=chunk_words)

    all_qs = []
    for idx, chunk in enumerate(chunks, start=1):
        with st.spinner(f"Generating questions for chunk {idx}/{len(chunks)}..."):
            # build a clear structured prompt
            prompt = (
                "Generate a diverse set of clear questions (WH, yes/no, explanation, reasoning) "
                "from the passage below. Return only the questions, one per line.\n\n"
                f"{chunk}"
            )
            # tune generation params based on model (long models can handle fewer beams if needed)
            try:
                # num_return_sequences and num_beams are set from per_chunk
                outs = qg_pipeline(
                    prompt,
                    max_length=256,
                    num_return_sequences=per_chunk,
                    num_beams=per_chunk if per_chunk > 1 else 1,
                    do_sample=True if per_chunk > 1 else False,
                    early_stopping=True,
                    clean_up_tokenization_spaces=True,
                )
            except Exception as e:
                st.warning(f"QG failed on chunk {idx}: {e}")
                continue

            # pipeline returns list of dicts
            for o in outs:
                text_out = o.get("generated_text", "")
                # split lines in case model returned multiple Qs in one output
                for line in text_out.splitlines():
                    q = normalize_q(line)
                    if q:
                        all_qs.append(q)

    # deduplicate preserving order (case-insensitive)
    seen = set()
    deduped = []
    for q in all_qs:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    return deduped

# ────────────────────────────────────────────────────────────────
# 3️⃣ Load QG model & tokenizer
with st.spinner("Loading question-generation model..."):
    qg_pipeline, qg_tokenizer, qg_model_name = load_qg_model(model_choice)
    st.write(f"Using model: `{qg_model_name}` (device: {'GPU' if _has_cuda else 'CPU'})")

# If user picked a long model and auto_chunk is checked, keep it but notify
if ("long-t5" in qg_model_name.lower() or "led" in qg_model_name.lower()) and auto_chunk:
    st.info("You selected a long-context model (LongT5 / LED). Auto-chunking is enabled — set Auto-chunk OFF if you want single-shot generation for very long documents.")

# ────────────────────────────────────────────────────────────────
# Generate questions (chunk-aware)
with st.spinner("Preparing question generation..."):
    questions_generated = generate_questions_from_text(
        qg_pipeline=qg_pipeline,
        qg_tokenizer=qg_tokenizer,
        text=translated_text,
        model_name=qg_model_name,
        auto_chunk=auto_chunk,
        chunk_words=chunk_words,
        token_chunk=chunk_by_tokens,
        per_chunk=num_qs_per_chunk,
    )

# Apply question-type filter
def filter_by_type(q):
    qlow = q.strip().lower()
    if not qlow.endswith("?"):
        qlow += "?"
    if not question_types:
        return True
    return any(qlow.startswith(qt.lower()) for qt in question_types)

filtered = [q for q in questions_generated if filter_by_type(q)]

if not filtered:
    st.warning("No questions matched the selected types after filtering. Showing all generated (pre-filter) questions.")
    filtered = questions_generated

# Display generated questions
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
