import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import spacy
import numpy as np
import librosa
import soundfile as sf
import tempfile
from pathlib import Path

# Optional: Speech recognition
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

# Optional: Audio recorder
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

# --- CONFIG ---
st.set_page_config(
    page_title="ABSA Voice Emotion & Sentiment Analyzer",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- CUSTOM UI STYLING ---
st.markdown("""
<style>

    /* --- MAIN BACKGROUND (Light Green) --- */
    .stApp {
        background: linear-gradient(135deg, #c8f7c5 0%, #a3e4a6 100%);
    }

    /* --- MAIN WHITE CARD (Container) --- */
    .main .block-container {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    /* --- HEADER --- */
    .main-header {
        font-size: 2.7rem;
        text-align: center;
        color: #1d3557;
        font-weight: bold;
        margin-bottom: 1.5rem;
    }

    /* --- TEXT AREAS (White + Black Text) --- */
    .stTextArea textarea {
        background: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #6abf69 !important;
        border-radius: 10px !important;
        padding: 15px !important;
        font-size: 1.1rem !important;
    }

    .stTextArea textarea:focus {
        border-color: #4caf50 !important;
        box-shadow: 0 0 8px rgba(76, 175, 80, 0.35) !important;
    }

    /* --- FILE UPLOADER --- */
    .uploadedFile {
        background: #ffffff !important;
        border: 2px solid #6abf69 !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }

    /* --- BUTTONS (Green Theme) --- */
    .stButton > button {
        background: linear-gradient(135deg, #4caf50 0%, #43a047 100%) !important;
        border: 2px solid #2e7d32 !important;
        color: white !important;
        padding: 0.5rem 2rem !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        transition: 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 14px rgba(76, 175, 80, 0.4) !important;
    }

    /* --- TABS --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #e8f5e9;
        border-radius: 10px 10px 0 0;
        padding: 8px 15px;
        border: 2px solid #81c784;
        color: #1b4332;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4caf50 0%, #43a047 100%);
        color: white !important;
        border-bottom: none;
    }

    /* --- RESULT CARDS --- */
    .result-card {
        background: #f1f8e9;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #66bb6a;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }

    .emotion-card { border-left-color: #43a047; }
    .transcript-card { border-left-color: #2e7d32; }
    .aspect-card { border-left-color: #66bb6a; }
    .final-card { border-left-color: #1b5e20; background: #1b5e20; color: white; }

    /* --- TAGS --- */
    .aspect-tag {
        background: #4caf50;
        color: white;
        padding: 7px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-right: 8px;
        display: inline-block;
        font-weight: bold;
    }

</style>
""", unsafe_allow_html=True)


# --- MODEL PATHS ---
ROBERTA_MODEL_PATH = "./Model/RoBERTa_ABSA_Final"
EMOTION_MODEL_PATH = "./Model/wav2vec2-ser-ravdess"
FALLBACK_MODEL = "roberta-base"

EMOTION_LABELS = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad", 
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    tokenizer = model = nlp = emotion_processor = emotion_model = None

    # RoBERTa
    try:
        model_path = Path(ROBERTA_MODEL_PATH)
        if model_path.exists() and any(model_path.glob("pytorch_model*.bin")):
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)
    except:
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)

    # Emotion model
    try:
        emotion_path = Path(EMOTION_MODEL_PATH)
        if emotion_path.exists():
            emotion_processor = Wav2Vec2Processor.from_pretrained(str(emotion_path))
            emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(str(emotion_path))
    except:
        pass

    # spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    return tokenizer, model, nlp, emotion_processor, emotion_model

tokenizer, model, nlp, emotion_processor, emotion_model = load_models()

# --- HELPER FUNCTIONS ---
def predict_sentiment(text):
    if not text.strip():
        return "Neutral", 0.5
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        conf = torch.max(probs).item()
        label = model.config.id2label.get(logits.argmax().item(), "Neutral")
        return label, conf
    except:
        return "Neutral", 0.5

def extract_aspects(text):
    if not nlp or not text.strip():
        return []
    doc = nlp(text)
    aspects = [chunk.text for chunk in doc.noun_chunks if not chunk.root.is_stop]
    return list(set([a.strip() for a in aspects if len(a.strip())>2]))

def detect_emotion(audio, sr):
    if not emotion_processor or not emotion_model:
        return "neutral"
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    inputs = emotion_processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    pred = logits.argmax().item()
    return EMOTION_LABELS.get(pred, "neutral")

def transcribe_audio(audio, sr):
    if not SPEECH_AVAILABLE:
        return "Speech recognition unavailable."
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio, sr)
            tmp_path = tmp.name
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio_rec = r.record(source)
            text = r.recognize_google(audio_rec)
        return text
    except:
        return "Could not transcribe audio."

def process_uploaded_file(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.read())
            path = tmp.name
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        return audio, sr
    except Exception as e:
        st.error(f"Audio processing failed: {e}")
        return None, None

# --- APP STATE ---
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "audio_sr" not in st.session_state:
    st.session_state.audio_sr = None
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "emotion" not in st.session_state:
    st.session_state.emotion = None

# --- UI ---
st.markdown('<h1 style="text-align:center;">🎤 ABSA Voice Emotion & Sentiment Analyzer</h1>', unsafe_allow_html=True)

tabs = st.tabs(["📝 Text Input", "🎤 Live Voice Recording", "📁 Upload Audio File"])
input_text = ""

# --- TEXT INPUT TAB ---
with tabs[0]:
    text = st.text_area("Enter your review or feedback:", placeholder="Example: Your food is disgusting!")
    if st.button("🔍 Analyze Text"):
        input_text = text.strip()

# --- LIVE VOICE RECORDING TAB (FIXED) ---
with tabs[1]:
    if AUDIO_RECORDER_AVAILABLE:
        audio_bytes = audio_recorder()
        if audio_bytes:

            # --- FIX: write bytes into a temp wav file ---
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # Read audio correctly
            audio, sr = sf.read(tmp_path)

            # Convert stereo to mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Save
            st.session_state.audio_data = audio
            st.session_state.audio_sr = sr

            # Run detection
            st.session_state.emotion = detect_emotion(audio, sr)
            st.session_state.transcript = transcribe_audio(audio, sr)
            input_text = st.session_state.transcript

            st.success(f"🎯 Transcribed: {input_text}")
            st.info(f"Detected Emotion: {st.session_state.emotion}")
    else:
        st.warning("Audio recorder not installed. Install `audio_recorder_streamlit` package.")

# --- UPLOAD FILE TAB ---
with tabs[2]:
    uploaded_file = st.file_uploader("Upload audio file (wav, mp3, m4a, flac):", type=["wav","mp3","m4a","flac"])
    if uploaded_file:
        st.session_state.audio_data, st.session_state.audio_sr = process_uploaded_file(uploaded_file)
        if st.session_state.audio_data is not None:
            st.session_state.emotion = detect_emotion(st.session_state.audio_data, st.session_state.audio_sr)
            st.session_state.transcript = transcribe_audio(st.session_state.audio_data, st.session_state.audio_sr)
            input_text = st.session_state.transcript
            st.success(f"🎯 Transcribed: {input_text}")
            st.info(f"Detected Emotion: {st.session_state.emotion}")
            st.audio(uploaded_file)

# --- ANALYSIS SECTION ---
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
