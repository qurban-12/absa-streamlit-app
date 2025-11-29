# ==============================
# ABSA Voice Emotion & Sentiment Analyzer
# Fully Commented Version for FYP/Presentation
# ==============================

# Import necessary libraries
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import spacy
import numpy as np
import librosa
import soundfile as sf
import tempfile
from pathlib import Path

# Optional packages for speech recognition and live audio
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True  # Flag to indicate speech recognition is available
except ImportError:
    SPEECH_AVAILABLE = False

try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True  # Flag to indicate live audio recorder is available
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

# ==============================
# STREAMLIT CONFIGURATION
# ==============================
st.set_page_config(
    page_title="ABSA Voice Emotion & Sentiment Analyzer",  # App title
    page_icon="🎤",                                         # Icon in browser tab
    layout="wide",                                         # Wide layout for better UI
)
st.markdown('<h1 style="text-align:center;">🎤 ABSA Voice Emotion & Sentiment Analyzer</h1>', unsafe_allow_html=True)

# ==============================
# MODEL PATHS & LABELS
# ==============================
ROBERTA_PATH = "./Model/RoBERTa_ABSA_Final"  # Path to custom RoBERTa model for sentiment/aspects
EMOTION_PATH = "./Model/wav2vec2-ser-ravdess"  # Path to wav2vec2 emotion recognition model
FALLBACK = "roberta-base"  # Fallback HuggingFace model if custom model not found

# Emotion labels mapping (index to emotion)
EMO_LABELS = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}

# ==============================
# MODEL LOADING FUNCTION
# ==============================
@st.cache_resource  # Cache to avoid reloading models on every run
def load_models():
    """
    Loads all AI models: sentiment (RoBERTa), emotion (wav2vec2), and NLP (spaCy).
    Returns tokenizer, sentiment model, spaCy NLP, emotion processor, emotion model.
    """
    # --- Load RoBERTa Sentiment Model ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_PATH)
    except:
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK)
        model = AutoModelForSequenceClassification.from_pretrained(FALLBACK)

    # --- Load Wav2Vec2 Emotion Model ---
    try:
        proc = Wav2Vec2Processor.from_pretrained(EMOTION_PATH)
        emo_model = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_PATH)
    except:
        proc = None
        emo_model = None

    # --- Load spaCy NLP Model for aspect extraction ---
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    return tokenizer, model, nlp, proc, emo_model

# Load all models
tokenizer, model, nlp, emo_proc, emo_model = load_models()

# ==============================
# HELPER FUNCTIONS
# ==============================

def predict_sentiment(text):
    """
    Predicts sentiment of input text using RoBERTa.
    Returns sentiment label and confidence score.
    """
    if not text.strip():
        return "Neutral", 0.5
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    label = model.config.id2label.get(logits.argmax().item(), "Neutral")
    confidence = float(torch.max(probs))
    return label, confidence

def extract_aspects(text):
    """
    Extracts aspects (noun phrases) from input text using spaCy.
    Returns a list of aspects.
    """
    if not nlp or not text.strip():
        return []
    doc = nlp(text)
    # Filter out short or stopword aspects
    aspects = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
    return list(set(aspects))  # Remove duplicates

def detect_emotion(audio, sr):
    """
    Detects emotion from audio data using Wav2Vec2 model.
    Audio is resampled to 16kHz if needed.
    Returns predicted emotion label.
    """
    if not emo_proc or not emo_model:
        return "neutral"
    if sr != 16000:
        audio = librosa.resample(audio, sr, 16000)
        sr = 16000
    inputs = emo_proc(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = emo_model(**inputs).logits
    pred_idx = logits.argmax().item()
    return EMO_LABELS.get(pred_idx, "neutral")

def transcribe_audio(audio, sr):
    """
    Transcribes audio to text using Google Speech Recognition.
    Returns transcript string.
    """
    if not SPEECH_AVAILABLE:
        return "Speech recognition unavailable."
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio, sr)
            r = sr.Recognizer()
            with sr.AudioFile(tmp.name) as source:
                audio_rec = r.record(source)
                return r.recognize_google(audio_rec)
    except:
        return "Could not transcribe audio."

def process_uploaded(file):
    """
    Reads uploaded audio file, converts to mono float32, resamples to 16kHz.
    Returns audio array and sample rate.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.read())
        path = tmp.name
    audio, sr = sf.read(path)
    if audio.ndim > 1:  # Stereo to mono
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, sr, 16000)
        sr = 16000
    return audio, sr

# ==============================
# STREAMLIT SESSION STATE
# ==============================
for key in ["audio_data", "audio_sr", "transcript", "emotion"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ==============================
# UI TABS
# ==============================
tabs = st.tabs(["📝 Text Input", "🎤 Live Voice Recording", "📁 Upload Audio File"])
input_text = ""

# ------------------------------
# TAB 1: Text Input
# ------------------------------
with tabs[0]:
    text = st.text_area("Enter your review or feedback:")
    if st.button("🔍 Analyze Text"):
        input_text = text.strip()

# ------------------------------
# TAB 2: Live Voice Recording
# ------------------------------
with tabs[1]:
    if AUDIO_RECORDER_AVAILABLE:
        audio_bytes = audio_recorder()
        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                path = tmp.name
            audio, sr = sf.read(path)
            if audio.ndim > 1: audio = audio.mean(axis=1)
            if audio.dtype != np.float32: audio = audio.astype(np.float32)
            st.session_state.audio_data = audio
            st.session_state.audio_sr = sr
            st.session_state.emotion = detect_emotion(audio, sr)
            st.session_state.transcript = transcribe_audio(audio, sr)
            input_text = st.session_state.transcript
            st.success(f"🎯 Transcribed: {input_text}")
            st.info(f"Detected Emotion: {st.session_state.emotion}")
    else:
        st.warning("Audio recorder not installed. Install `audio_recorder_streamlit`.")

# ------------------------------
# TAB 3: Upload Audio File
# ------------------------------
with tabs[2]:
    uploaded_file = st.file_uploader("Upload audio (wav, mp3, m4a, flac):", type=["wav","mp3","m4a","flac"])
    if uploaded_file:
        audio, sr = process_uploaded(uploaded_file)
        st.session_state.audio_data = audio
        st.session_state.audio_sr = sr
        st.session_state.emotion = detect_emotion(audio, sr)
        st.session_state.transcript = transcribe_audio(audio, sr)
        input_text = st.session_state.transcript
        st.success(f"🎯 Transcribed: {input_text}")
        st.info(f"Detected Emotion: {st.session_state.emotion}")
        st.audio(uploaded_file)

# ------------------------------
# ANALYSIS SECTION
# ------------------------------
if input_text:
    st.divider()
    st.markdown("### 📊 Analysis Results")
    aspects = extract_aspects(input_text)
    sentiment, conf = predict_sentiment(input_text)
    if st.session_state.emotion:
        st.info(f"Detected Emotion: {st.session_state.emotion}")
    st.info(f"Predicted Sentiment: {sentiment} (Confidence: {conf:.2f})")
    if aspects:
        st.info(f"Detected Aspects: {', '.join(aspects)}")
    final_out = f"Customer is {st.session_state.emotion.lower() if st.session_state.emotion else sentiment.lower()}"
    if aspects:
        final_out += f" about: {', '.join(aspects)}"
    st.success(f"🚀 FINAL OUTPUT: {final_out}")
else:
    st.info("👆 Enter text, record live audio, or upload audio to start analysis.")
