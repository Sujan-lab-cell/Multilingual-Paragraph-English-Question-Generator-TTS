# streamlit_nlp_qg_app.py
# Streamlit app: Audio/Text -> Translate -> Question Generation -> TTS (bytes-based TTS)
# Patched: use st.session_state + io.BytesIO to reliably play audio after button clicks

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from deep_translator import GoogleTranslator
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS
import tempfile
import os
import io
import nltk
from langdetect import detect
from nltk.tokenize import sent_tokenize

# Ensure required NLTK data is available (download only if needed)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

st.set_page_config(page_title="NLP QG App", layout="centered")
st.title("📚 NLP: Audio/Text → Translate → Question Generation → TTS")
st.markdown(
    "This app accepts audio or text (500+ words recommended), auto-detects language, translates to English if needed, "
    "generates questions from the text and provides TTS for the input text and each generated question.\n\n"
    "TTS is generated with gTTS and played as in-memory bytes (survives Streamlit reruns)."
)

# Sidebar settings
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox(
    "Question generation model",
    options=[
        "valhalla/t5-small-qg-hl",
        "mrm8488/t5-base-finetuned-question-generation"
    ],
)
max_questions = st.sidebar.slider("Max questions", min_value=5, max_value=50, value=15)
allow_short = st.sidebar.checkbox("Allow less than 500 words", value=False)

# Initialize model & tokenizer (lazy load)
@st.cache_resource
def load_qg_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    device = -1
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        try:
            device = int(os.environ.get('CUDA_VISIBLE_DEVICES').split(',')[0])
        except Exception:
            device = 0
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return pipe

with st.spinner("Loading question-generation model (can take a minute on first run)..."):
    qg_pipe = load_qg_model(model_name)

# ---------------- Helpers ----------------
def transcribe_audio(uploaded_file):
    # Accept wav, mp3, m4a, ogg
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tf:
        tf.write(uploaded_file.read())
        tmp_path = tf.name

    wav_path = tmp_path + ".wav"
    try:
        audio = AudioSegment.from_file(tmp_path)
        audio.export(wav_path, format="wav")
    except Exception as e:
        st.error(f"Audio conversion error: {e}")
        try:
            os.remove(tmp_path)
        except:
            pass
        return ""

    r = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = ""
    except Exception as e:
        st.error(f"Speech recognition error: {e}")
        text = ""

    # cleanup
    try:
        os.remove(tmp_path)
        os.remove(wav_path)
    except:
        pass
    return text

def translate_to_english(text):
    try:
        detected = detect(text)
    except Exception:
        detected = 'en'
    if detected == 'en':
        return text, detected
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated, detected
    except Exception:
        return text, detected

@st.cache_data
def generate_tts_bytes(text, lang='en'):
    """
    Generate MP3 bytes for given text using gTTS, cached.
    Returns raw bytes or None on failure.
    """
    if not text or text.strip() == "":
        return None
    try:
        # save to temp file (gTTS requires saving), read bytes, then delete file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_name = tmp_file.name
        tmp_file.close()
        tts = gTTS(text=text, lang=lang)
        tts.save(tmp_name)
        with open(tmp_name, "rb") as f:
            audio_bytes = f.read()
        try:
            os.remove(tmp_name)
        except Exception:
            pass
        return audio_bytes
    except Exception as e:
        # don't call st.error inside a cached function in heavy use, keep simple return
        return None

def generate_questions_from_text(text, max_q=10):
    # naive approach: split into sentences and highlight each sentence to create Qs
    sents = sent_tokenize(text)
    questions = []
    attempts = 0
    used = set()
    for i, sent in enumerate(sents):
        if len(questions) >= max_q:
            break
        try:
            context = text
            hl_input = context.replace(sent, f"<hl> {sent} <hl>")
            prompt = f"generate question: {hl_input}"
            res = qg_pipe(prompt, max_length=64, num_return_sequences=1)
            # handle different pipeline output formats
            if isinstance(res, list) and len(res) > 0:
                out = res[0]
                qtxt = out.get('generated_text') or out.get('text') or str(out)
                qtxt = qtxt.strip()
            else:
                qtxt = ""
            # normalize and deduplicate
            norm = qtxt.strip().lower().strip(' .?!"\'')
            if qtxt and norm not in used and qtxt.endswith('?'):
                questions.append(qtxt)
                used.add(norm)
        except Exception:
            attempts += 1
            continue
    # fallback: if no questions generated, simple heuristic questions
    if not questions:
        for s in sents[:max_q]:
            if len(questions) >= max_q:
                break
            q = None
            if ' is ' in s:
                part = s.split(' is ', 1)[0]
                q = f"What is {part.strip()}?"
            else:
                q = f"Can you explain: {s[:60].strip()}?"
            norm = q.strip().lower().strip(' .?!"\'')
            if norm not in used:
                questions.append(q)
                used.add(norm)
    return questions

# Helper to generate and store audio bytes in session_state
def play_text_as_audio_ss(ss_key, text, lang='en'):
    """
    Generate TTS bytes and store under st.session_state[ss_key].
    Returns True if available.
    """
    if not text or text.strip() == "":
        return False
    # If already generated, return True if not None
    if st.session_state.get(ss_key):
        return True
    audio_bytes = generate_tts_bytes(text, lang=lang)
    if audio_bytes:
        st.session_state[ss_key] = audio_bytes
        return True
    else:
        st.session_state[ss_key] = None
        return False

# initialize session_state keys if not present (avoids KeyError)
if 'translated_audio' not in st.session_state:
    st.session_state['translated_audio'] = None
if 'input_audio' not in st.session_state:
    st.session_state['input_audio'] = None

# ---------------- UI ----------------
st.header("Input")
input_mode = st.radio("Input type:", ['Paste text', 'Upload audio'])

input_text = ''
if input_mode == 'Paste text':
    input_text = st.text_area("Paste or type your text here (500+ words recommended)", height=250)
else:
    audio_file = st.file_uploader("Upload an audio file (wav/mp3/m4a/ogg)", type=['wav','mp3','m4a','ogg'])
    if audio_file is not None:
        with st.spinner("Transcribing audio..."):
            transcribed = transcribe_audio(audio_file)
        if not transcribed:
            st.warning("Could not transcribe audio or audio was empty.")
        input_text = st.text_area("Transcribed text (editable)", value=transcribed, height=250)

process = st.button("Analyze & Generate Questions")

if process:
    if not input_text or (len(input_text.split()) < 500 and not allow_short):
        st.error("Please provide at least 500 words (or enable 'Allow less than 500 words' in sidebar).")
    else:
        st.info("Detecting language and translating if necessary...")
        with st.spinner("Detecting language..."):
            translated, detected = translate_to_english(input_text)

        st.write("---")
        if detected != 'en':
            st.success(f"Detected language: {detected}. Translated to English for analysis.")
            st.subheader("Translated text (English)")
            st.write(translated)
            # Play translated text: generate into session_state on button press, render audio after
            if st.button("🔊 Play translated text", key="play_translated_btn"):
                ok = play_text_as_audio_ss("translated_audio", translated, lang='en')
                if not ok:
                    st.warning("TTS generation failed or no audio returned.")
        else:
            st.success("Language: English (no translation needed).")
            translated = input_text
            st.subheader("Input text (English)")
            st.write(translated)
            # Play input text: generate into session_state on button press, render audio after
            if st.button("🔊 Play input text", key="play_input_btn"):
                ok = play_text_as_audio_ss("input_audio", translated, lang='en')
                if not ok:
                    st.warning("TTS generation failed or no audio returned.")

        # Render audio players (outside button branches so they persist across reruns)
        if st.session_state.get("translated_audio"):
            try:
                st.audio(io.BytesIO(st.session_state["translated_audio"]), format="audio/mp3")
            except Exception:
                st.warning("Unable to play translated audio in this environment.")
        if st.session_state.get("input_audio"):
            try:
                st.audio(io.BytesIO(st.session_state["input_audio"]), format="audio/mp3")
            except Exception:
                st.warning("Unable to play input audio in this environment.")

        st.header("Generated Questions")
        with st.spinner("Generating questions..."):
            questions = generate_questions_from_text(translated, max_q=max_questions)

        if questions:
            st.write(f"Generated {len(questions)} questions:")
            # Play all questions (joined with line breaks) - stores under 'all_questions_audio'
            if st.button("🔊 Play all questions", key="play_all_qs_btn"):
                all_q = '\n'.join([f"{i+1}. {q}" for i,q in enumerate(questions)])
                ok = play_text_as_audio_ss("all_questions_audio", all_q, lang='en')
                if not ok:
                    st.warning("TTS generation failed for all questions.")
            # Render audio for all questions if available
            if st.session_state.get("all_questions_audio"):
                try:
                    st.audio(io.BytesIO(st.session_state["all_questions_audio"]), format="audio/mp3")
                except Exception:
                    st.warning("Unable to play bulk questions audio in this environment.")

            # Show each question with an inline play button
            for i, q in enumerate(questions, 1):
                cols = st.columns([0.9, 0.1])
                with cols[0]:
                    st.markdown(f"**{i}. {q}**")
                with cols[1]:
                    play_key = f"play_q_{i}_btn"
                    ss_key = f"q_audio_{i}"
                    if st.button("🔊", key=play_key):
                        ok = play_text_as_audio_ss(ss_key, q, lang='en')
                        if not ok:
                            st.warning(f"TTS generation failed for question {i}.")
                # render audio below (or inline) if it's generated
                if st.session_state.get(ss_key):
                    try:
                        st.audio(io.BytesIO(st.session_state[ss_key]), format="audio/mp3")
                    except Exception:
                        st.warning(f"Unable to play audio for question {i} in this environment.")

            # Bulk TTS & download
            st.subheader("Bulk TTS & Download")
            if st.button("Generate TTS for input text (bulk)", key="bulk_input_tts_btn"):
                ok = play_text_as_audio_ss("bulk_input_audio", translated, lang='en')
                if not ok:
                    st.warning("Bulk TTS failed for input text.")
            if st.session_state.get("bulk_input_audio"):
                try:
                    st.audio(io.BytesIO(st.session_state["bulk_input_audio"]), format="audio/mp3")
                except Exception:
                    st.warning("Unable to play bulk input audio in this environment.")

            if st.button("Generate TTS for questions (bulk)", key="bulk_q_tts_btn"):
                all_q = '\n'.join([f"{i+1}. {q}" for i,q in enumerate(questions)])
                ok = play_text_as_audio_ss("bulk_q_audio", all_q, lang='en')
                if not ok:
                    st.warning("Bulk TTS failed for questions.")
            if st.session_state.get("bulk_q_audio"):
                try:
                    st.audio(io.BytesIO(st.session_state["bulk_q_audio"]), format="audio/mp3")
                except Exception:
                    st.warning("Unable to play bulk questions audio in this environment.")

            # Allow download of questions as txt
            if st.download_button("Download questions (.txt)", data='\n'.join(questions), file_name='questions.txt'):
                st.success("Downloaded!")
        else:
            st.warning("No questions were generated. Try a longer or clearer text, or change the model in settings.")

st.markdown("---")
st.caption("Tip: For better question generation, provide well-formed paragraphs of factual text ( > 500 words ).")
# End of file
