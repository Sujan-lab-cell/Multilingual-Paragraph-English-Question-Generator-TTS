# streamlit_nlp_qg_app.py
# Streamlit app: Audio/Text -> Translate -> Question Generation -> TTS
# Notes:
# - Uses transformers for question generation (T5 model).
# - Uses deep_translator for translation when input is non-English.
# - Uses speech_recognition + pydub to transcribe uploaded audio files.
# - Uses gTTS for Text-to-Speech output (mp3) and Streamlit to play it.

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
    "This app accepts audio or text (500+ words recommended), auto-detects language, translates to English if needed, generates questions from the text and provides TTS for original text and generated questions.\n\nNow with inline TTS players for the input text and each generated question."
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
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if (os.environ.get('CUDA_VISIBLE_DEVICES')) else -1)
    return pipe

with st.spinner("Loading question-generation model (can take a minute)..."):
    qg_pipe = load_qg_model(model_name)

# Helpers

def transcribe_audio(uploaded_file):
    # Accept wav, mp3, m4a, ogg
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tf:
        tf.write(uploaded_file.read())
        tmp_path = tf.name

    # convert to wav for recognizer
    wav_path = tmp_path + ".wav"
    try:
        audio = AudioSegment.from_file(tmp_path)
        audio.export(wav_path, format="wav")
    except Exception as e:
        st.error(f"Audio conversion error: {e}")
        return ""

    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = r.record(source)
    try:
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
        # fallback: return original
        return text, detected


@st.cache_data
def generate_tts_file(text, lang='en', filename_prefix='tts'):
    """Generate TTS mp3 and cache result to avoid re-generation."""
    try:
        tts = gTTS(text=text, lang=lang)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", prefix=filename_prefix)
        tts.save(tmp.name)
        return tmp.name
    except Exception as e:
        st.error(f"TTS generation failed: {e}")
        return None


def generate_questions_from_text(text, max_q=10):
    # naive approach: split into sentences and highlight each sentence to create Qs
    sents = sent_tokenize(text)
    questions = []
    attempts = 0
    for i, sent in enumerate(sents):
        if len(questions) >= max_q:
            break
        try:
            context = text
            hl_input = context.replace(sent, f"<hl> {sent} <hl>")
            prompt = f"generate question: {hl_input}"
            res = qg_pipe(prompt, max_length=64, num_return_sequences=1)
            qtxt = res[0]['generated_text'].strip()
            if qtxt and qtxt not in questions and qtxt.endswith('?'):
                questions.append(qtxt)
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
            questions.append(q)
    return questions

# Main UI inputs
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

# Buttons
process = st.button("Analyze & Generate Questions")

if process:
    if not input_text or (len(input_text.split()) < 500 and not allow_short):
        st.error("Please provide at least 500 words (or enable 'Allow less than 500 words' in sidebar).")
    else:
        st.info("Detecting language and translating if necessary...")
        with st.spinner("Detecting language..."):
            translated, detected = translate_to_english(input_text)
        if detected != 'en':
            st.success(f"Detected language: {detected}. Translated to English for analysis.")
            st.write("---")
            st.subheader("Translated text (English)")
            st.write(translated)
            # Inline TTS player for translated text
            if st.button("🔊 Play translated text", key="play_translated"):
                tts_path = generate_tts_file(translated, lang='en', filename_prefix='input_text_')
                if tts_path:
                    st.audio(tts_path)
        else:
            st.success("Language: English (no translation needed).")
            translated = input_text
            st.write("---")
            st.subheader("Input text (English)")
            st.write(translated)
            # Inline TTS player for input text
            if st.button("🔊 Play input text", key="play_input"):
                tts_path = generate_tts_file(translated, lang='en', filename_prefix='input_text_')
                if tts_path:
                    st.audio(tts_path)

        st.header("Generated Questions")
        with st.spinner("Generating questions..."):
            questions = generate_questions_from_text(translated, max_q=max_questions)

        if questions:
            st.write(f"Generated {len(questions)} questions:")

            # Play all questions button
            if st.button("🔊 Play all questions", key="play_all_qs"):
                all_q = '\n'.join([f"{i+1}. {q}" for i,q in enumerate(questions)])
                tts_all_path = generate_tts_file(all_q, lang='en', filename_prefix='questions_all_')
                if tts_all_path:
                    st.audio(tts_all_path)

            # Show each question with an inline play button
            for i, q in enumerate(questions, 1):
                cols = st.columns([0.9, 0.1])
                with cols[0]:
                    st.markdown(f"**{i}. {q}**")
                with cols[1]:
                    play_key = f"play_q_{i}"
                    if st.button("🔊", key=play_key):
                        tts_q_path = generate_tts_file(q, lang='en', filename_prefix=f'question_{i}_')
                        if tts_q_path:
                            st.audio(tts_q_path)

            # Provide TTS options for bulk generation as before
            st.subheader("Bulk TTS & Download")
            if st.button("Generate TTS for input text (bulk)", key="bulk_input_tts"):
                tts_path = generate_tts_file(translated, filename_prefix='input_text_')
                if tts_path:
                    st.audio(tts_path)
            if st.button("Generate TTS for questions (bulk)", key="bulk_q_tts"):
                all_q = '\n'.join([f"{i+1}. {q}" for i,q in enumerate(questions)])
                tts_q_path = generate_tts_file(all_q, filename_prefix='questions_')
                if tts_q_path:
                    st.audio(tts_q_path)

            # Allow download of questions as txt
            if st.download_button("Download questions (.txt)", data='\n'.join(questions), file_name='questions.txt'):
                st.success("Downloaded!")
        else:
            st.warning("No questions were generated. Try a longer or clearer text, or change the model in settings.")

st.markdown("---")
st.caption("Tip: For better question generation, provide well-formed paragraphs of factual text ( > 500 words ).")


# End of file