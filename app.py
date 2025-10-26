import os
# Ensure FFmpeg path is correctly added if needed outside of standard PATH
# os.environ["PATH"] += os.pathsep + r"C:\Your\Path\To\ffmpeg\bin" # Example

from pydub import AudioSegment
from pydub.utils import which

# Try to automatically find ffmpeg/ffprobe, handle if not found
try:
    AudioSegment.converter = which("ffmpeg")
    AudioSegment.ffprobe = which("ffprobe")
    print("ffmpeg path:", AudioSegment.converter)
    print("ffprobe path:", AudioSegment.ffprobe)
    if not AudioSegment.converter or not AudioSegment.ffprobe:
        print("WARNING: ffmpeg or ffprobe not found in system PATH. Audio conversion might fail.")
        # Optionally, provide a manual path if which() fails
        # AudioSegment.converter = r"C:\Your\Path\To\ffmpeg\bin\ffmpeg.exe"
        # AudioSegment.ffprobe = r"C:\Your\Path\To\ffmpeg\bin\ffprobe.exe"
except Exception as e:
    print(f"Error setting up pydub converters: {e}")


from flask import Flask, render_template, request, redirect, url_for
import cv2
from speechbrain.pretrained.interfaces import foreign_class
import pickle
from transformers import pipeline
import pandas as pd # Make sure pandas is imported if text_results is a DataFrame
import nltk # Import nltk for stopwords download check
from nltk.corpus import stopwords
from functions import * # Make sure functions.py is in the same directory or accessible
import warnings
import traceback
import tensorflow as tf
import torch
from collections import Counter
import time # For unique filenames

warnings.filterwarnings("ignore")

app = Flask(__name__)
# Configure a directory for uploads
UPLOAD_FOLDER = 'uploads'
CONVERTED_FOLDER = 'converted_audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)


# --- Load Models and Constants (Consider loading within functions if memory is an issue) ---
try:
    stop_words = stopwords.words('english')
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

CLASSES_EMO_FACE = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# --- Make sure these files exist in your project directory ---
FACE_CASCADE_PATH = "haarcascade_face.xml"
EYE_CASCADE_PATH = "haarcascade_eye.xml"
FACE_EMOTION_MODEL_PATH = "model_mv1.h5"
SPEECH_MODEL_PATH = "emotion_model_29_01_2023.h5"
TOKENIZER_PATH = 'tokenizer.pkl'

if not os.path.exists(FACE_CASCADE_PATH): print(f"ERROR: Cannot find {FACE_CASCADE_PATH}")
if not os.path.exists(EYE_CASCADE_PATH): print(f"ERROR: Cannot find {EYE_CASCADE_PATH}")
if not os.path.exists(FACE_EMOTION_MODEL_PATH): print(f"ERROR: Cannot find {FACE_EMOTION_MODEL_PATH}")
if not os.path.exists(SPEECH_MODEL_PATH): print(f"ERROR: Cannot find {SPEECH_MODEL_PATH}")
if not os.path.exists(TOKENIZER_PATH): print(f"ERROR: Cannot find {TOKENIZER_PATH}")
# ---

FACE_CASCADE = cv2.CascadeClassifier(FACE_CASCADE_PATH)
EYE_CASCADE = cv2.CascadeClassifier(EYE_CASCADE_PATH)
FACE_EMOTION_MODEL = tf.keras.models.load_model(FACE_EMOTION_MODEL_PATH)
SPEECH_CLASSIFICATION = tf.keras.models.load_model(SPEECH_MODEL_PATH)

encoder = {0:"anger", 1:"fear", 2:"joy", 3:"love", 4:"sadness", 5:"surprise"}

RESIZE_SIZES = {"single":(50, 50), "double":(100, 50)}
RESIZE = True
FACE_PERC = 0.6
N_FPS = 1
SHOW = False
MODE = "rgb"
USE = "both" #either single or both

with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
    # Add savedir if needed to specify download location, e.g., savedir="pretrained_models"
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 30
# Consider specifying a cache_dir for Hugging Face models
# cache_dir = "./hf_cache"
# os.makedirs(cache_dir, exist_ok=True)
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny", # Or a larger model if accuracy is low
    chunk_length_s=CHUNK_SIZE,
    device=device,
    # cache_dir=cache_dir
)

print("Models loaded successfully.")

# --- Flask Routes ---

@app.route("/")
def index():
    """Serves the new Home page"""
    return render_template("index.html")

@app.route("/about")
def about():
    """Serves the About page"""
    return render_template("about.html")

@app.route("/contact")
def contact():
    """Serves the Contact page"""
    return render_template("contact.html")

@app.route("/analysis")
def analysis():
    """Serves the Analysis (video upload) page"""
    return render_template("analysis.html")


@app.route("/submit_video", methods=["POST"])
def process():
    video_file = request.files.get("video")

    if not video_file or video_file.filename == '':
        print("No video file found in request.")
        # Optionally add a flash message here for the user
        return redirect(url_for('analysis'))

    # Use a unique filename to avoid conflicts if multiple users access
    timestamp = str(int(time.time()))
    safe_filename = f"interview_{timestamp}.mp4" # Basic sanitization needed for real apps
    video_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    converted_audio_path = os.path.join(CONVERTED_FOLDER, f"audio_{timestamp}.wav")

    # Initialize cap outside try block for finally clause
    cap = None
    try:
        video_file.save(video_path)
        print(f"Video saved to {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            # Clean up saved file before redirecting
            if os.path.exists(video_path): os.remove(video_path)
            return redirect(url_for('analysis'))

        print("Starting face/eye/emotion analysis...")
        eyes_list, both_eyes, no_eyes, no_face, face_emotions = face_eye_emotion(
            video_path=video_path,
            face_perc=FACE_PERC, n_fps=N_FPS, resize=RESIZE, show=SHOW, mode=MODE,
            resize_sizes=RESIZE_SIZES, FACE_CASCADE=FACE_CASCADE, EYE_CASCADE=EYE_CASCADE,
            emotion_model=FACE_EMOTION_MODEL, CLASSES_EMO_FACE=CLASSES_EMO_FACE
        )
        print("Finished face/eye/emotion analysis.")
        cap.release() # Release video capture *after* analysis
        cap = None # Set to None after release

        print("Starting post-processing...")
        face_emotion_eye = post_process(
            eyes_list=eyes_list, both_eyes=both_eyes, use=USE,
            face_emotion=face_emotions, no_eyes=no_eyes, no_face=no_face
        )
        print("Finished post-processing.")

        print("Converting video to audio...")
        convert(video_path, converted_audio_path)
        print(f"Audio converted to {converted_audio_path}")

        print("Starting speech-to-text...")
        # Make sure speech2text_pipeline can accept audio_path or reads a default
        text_results = speech2text_pipeline(pipe, converted_audio_path) # Pass path explicitly
        print("Finished speech-to-text.")

        # --- Check structure of text_results before using ---
        if text_results is None or not isinstance(text_results, pd.DataFrame) or 'timestamp' not in text_results.columns:
             print(f"Error: speech2text_pipeline did not return expected DataFrame. Got: {type(text_results)}")
             # Handle error gracefully - maybe return results page with a message
             results = [] # Pass empty list
             sound_class = [] # Pass empty list
             # Or raise a specific error to be caught below if severe
             # raise ValueError("Speech-to-text processing failed or returned unexpected format.")
        else:
             print("Dividing audio clips...")
             # Check if timestamps are valid before passing
             if not text_results["timestamp"].empty:
                 divide_clips(text_results["timestamp"].values, converted_audio_path)
                 print("Finished dividing clips.")
             else:
                 print("No timestamps found in text_results, skipping divide_clips.")


             print("Making vocal inferences (classifier)...")
             # Ensure make_inferences knows where the clips are (might need path arg or reads from default location)
             sound_class = make_inferences(classifier)
             print("Finished making vocal inferences.")

             print("Classifying speech text...")
             results = classify_speech(text_results, SPEECH_CLASSIFICATION, tokenizer)
             print("Finished classifying speech text.")


        # --- CRITICAL DEBUGGING STEP ---
        print("-" * 20)
        results_type = type(results)
        print(f"Structure of 'results' before rendering: {results_type}")
        if isinstance(results, list) and len(results) > 0:
            first_item = results[0]
            first_item_type = type(first_item)
            print(f"First element of 'results': {first_item} (Type: {first_item_type})")
            # Check if it matches the expected dict structure {'text': 'emotion'}
            if not (isinstance(first_item, dict) and len(first_item) == 1):
                 print("WARNING: Structure DOES NOT match {'text': 'emotion'} dictionary!")
                 print("Loop in analyzed.html might fail or show incorrect data.")
            else:
                 print("Structure appears to match {'text': 'emotion'} dictionary.")

        elif isinstance(results, list):
             print("'results' is an empty list.")
        else:
             print(f"'results' is not a list or None: {results}")
             results = [] # Default to empty list for template
        print("-" * 20)
        # --- END DEBUGGING ---


        # Ensure sound_class is a list before passing to Counter
        if not isinstance(sound_class, list):
            print(f"Warning: sound_class is not a list (Type: {type(sound_class)}). Setting emotion counts to 0.")
            sound_class = [] # Default to empty list
        emotion_counts = Counter(sound_class) # Count vocal emotions

        print("Rendering results page...")
        # --- *** THIS LINE MUST POINT TO analyzed.html *** ---
        return render_template(
            "analyzed.html", # <-- RENDER THE RESULTS PAGE
            # --- Pass data safely using .get() with defaults ---
            neutral_face=face_emotion_eye.get("Emotion data", {}).get("Neutral", 0),
            happy_face=face_emotion_eye.get("Emotion data", {}).get("Happy", 0),
            pay_attention=face_emotion_eye.get("Attention %", 0.0), # Use 0.0 for float default
            neutral=emotion_counts.get("neutral", 0),
            happy=emotion_counts.get("happy", 0),
            sad=emotion_counts.get("sad", 0),
            text_data=results # Already ensured results is a list
        )

    except Exception as exc:
        # Ensure cap is released if it was opened
        if cap and cap.isOpened():
             cap.release()
             print("Video capture released due to exception.")
        print(f"Error during processing: {traceback.format_exc()}")
        # Render a user-friendly error page
        return render_template("error.html", error_message=str(exc), traceback_info=traceback.format_exc()), 500
    finally:
        # Clean up files regardless of success or failure
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Removed video file: {video_path}")
            except PermissionError:
                print(f"Could not delete {video_path} - possibly still locked.")
            except Exception as e:
                 print(f"Error removing video file {video_path}: {e}")

        if os.path.exists(converted_audio_path):
            try:
                # Add delay before removing audio if needed
                # time.sleep(1)
                os.remove(converted_audio_path)
                print(f"Removed audio file: {converted_audio_path}")
            except PermissionError:
                 print(f"Could not delete {converted_audio_path} - possibly still locked.")
            except Exception as e:
                 print(f"Error removing audio file {converted_audio_path}: {e}")

            # Also remove clip files if 'divide_clips' creates them in CONVERTED_FOLDER
            try:
                clip_prefix = f"clip_{timestamp}"
                for f in os.listdir(CONVERTED_FOLDER):
                    if f.startswith(clip_prefix):
                         clip_path = os.path.join(CONVERTED_FOLDER, f)
                         try:
                             os.remove(clip_path)
                             # print(f"Removed clip file: {clip_path}") # Optional: uncomment for verbose logging
                         except Exception as e_clip:
                             print(f"Could not remove clip {clip_path}: {e_clip}")
            except Exception as e_listdir:
                 print(f"Error listing directory {CONVERTED_FOLDER} for clip cleanup: {e_listdir}")


if __name__ == "__main__":
    # Consider using waitress or gunicorn for production instead of Flask's dev server
    app.run(host="localhost", debug=True) # debug=False for production

