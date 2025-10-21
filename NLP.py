# app.py
import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langdetect import detect
from gtts import gTTS
import tempfile, os
from pydub import AudioSegment
import speech_recognition as sr

st.set_page_config(page_title="Multilingual QG", layout="wide")
st.title("Multilingual Paragraph → English + Question Generator + TTS")

# Sidebar options
st.sidebar.header("Options")
use_speech = st.sidebar.checkbox("Use microphone (speech input)", value=False)
question_types = st.sidebar.multiselect("Allowed question types", 
                                        ["Who", "What", "When", "Where", "Why", "How", "Which"], 
                                        default=["Who","What","When","Where","Why","How"])
tts_engine = st.sidebar.selectbox("TTS engine", ["gTTS", "pyttsx3"], index=0)

# Load models lazily to keep startup time faster
@st.cache_resource(show_spinner=False)
def load_translation_model():
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

@st.cache_resource(show_spinner=False)
def load_qg_model():
    qg_model_name = "mrm8488/t5-base-finetuned-question-generation-ap"  # or valhalla/t5-small-qg-hl
    qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_name)
    qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model_name)
    return pipeline("text2text-generation", model=qg_model, tokenizer=qg_tokenizer, 
                    device=0 if (os.environ.get("CUDA_VISIBLE_DEVICES")) else -1)

# Input section
if use_speech:
    st.write("Record audio (single sentence/paragraph). Click Upload or Record locally and upload a WAV/MP3.")
    audio_file = st.file_uploader("Upload audio file (wav/mp3) or record externally and upload", type=["wav","mp3","m4a","ogg"])
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_file.getvalue())
            temp_audio_path = f.name
        # speech recognition
        r = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio = r.record(source)
        try:
            text = r.recognize_google(audio)  # small demo; replace with whisper for offline/high accuracy
            st.markdown("**Recognized text:**")
            st.write(text)
        except Exception as e:
            st.error(f"Speech-to-text error: {e}")
            st.stop()
else:
    text = st.text_area("Paste or type a paragraph in any language", height=200)

if not text or text.strip()=="":
    st.info("Enter text or upload audio to proceed.")
    st.stop()

# 1) Language detection
try:
    lang = detect(text)
except Exception:
    lang = "unknown"

st.write(f"Detected language code: **{lang}**")

# 2) Translation -> English if needed
translation_model, translation_tokenizer = load_translation_model()

translated_text = text
if lang != "en" and lang != "unknown":
    # M2M100 needs language id tokens set before encoding
    # map simple lang codes to m2m100 token (this is a minimal mapping; extend for production)
    m2m_lang_map = {
        "fr":"fr", "de":"de", "es":"es", "hi":"hi", "bn":"bn", "ar":"ar", "ru":"ru",
        "zh-cn":"zh", "zh":"zh", "pt":"pt", "it":"it"
    }
    src_lang = lang
    tgt_lang = "en"
    try:
        translation_tokenizer.src_lang = src_lang
        encoded = translation_tokenizer(text, return_tensors="pt")
        generated_tokens = translation_model.generate(**encoded, forced_bos_token_id=translation_tokenizer.get_lang_id(tgt_lang), max_length=512)
        translated_text = translation_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    except Exception as e:
        st.warning(f"Auto-translation using m2m100 failed: {e}. Using original text.")
        translated_text = text

st.markdown("**English (translated if needed):**")
st.write(translated_text)

# 3) Question Generation
qg_pipeline = load_qg_model()

# We'll craft a simple prompt: ask model to generate multiple questions from paragraph.
# You may tweak prompt formatting depending on model.
prompt = f"generate questions: {translated_text}"
# Many QG models generate one question. We can ask for X questions by specifying: "generate 5 questions"
num_qs = st.sidebar.slider("Number of questions to generate", 1, 10, 3)
prompt = f"generate {num_qs} questions: {translated_text}"

with st.spinner("Generating questions..."):
    out = qg_pipeline(prompt, max_length=256, num_return_sequences=1)[0]["generated_text"]

# out might be a list-like string; split heuristically into sentences/questions
import re
raw_qs = re.split(r"\n|(?<=[?])\s+", out.strip())
# filter by question type
def filter_by_type(q):
    t = q.strip().lower()
    if not t.endswith("?"):
        # if model output not ended with ?, add ? 
        t = t + "?"
    if not question_types:
        return True
    for qt in question_types:
        if t.startswith(qt.lower()):
            return True
    return False

questions = [q.strip() for q in raw_qs if q.strip() != ""]
filtered = [q for q in questions if filter_by_type(q)]

if len(filtered)==0:
    st.warning("No questions matched the selected types — relaxing filter to show all generated questions.")
    filtered = questions

st.markdown("### Generated questions")
for i, q in enumerate(filtered, start=1):
    st.write(f"{i}. {q}")

# 4) TTS for questions -> concatenate and play
if st.button("Play questions (TTS)"):
    combined = " ".join(filtered)
    if tts_engine == "gTTS":
        try:
            tts = gTTS(combined, lang="en")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp.name)
            st.audio(tmp.name)
        finally:
            pass
    else:
        # fallback: use pyttsx3 for local TTS (not always supported in Streamlit)
        import pyttsx3
        engine = pyttsx3.init()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        engine.save_to_file(combined, tmp.name)
        engine.runAndWait()
        st.audio(tmp.name)

st.success("Done")
