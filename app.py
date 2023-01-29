from flask import Flask
from flask import render_template, request, redirect
import cv2
from speechbrain.pretrained.interfaces import foreign_class
import pickle
from transformers import pipeline
from nltk.corpus import stopwords
from functions import *
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
app = Flask(__name__)
stop_words = stopwords.words('english')
CLASSES_EMO_FACE = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

FACE_CASCADE = cv2.CascadeClassifier("haarcascade_face.xml") 
FACE_EMOTION_MODEL_PATH = "model_mv1.h5"
EYE_CASCADE = cv2.CascadeClassifier("haarcascade_eye.xml")  # capture frames from a camera 
FACE_EMOTION_MODEL = tf.keras.models.load_model(FACE_EMOTION_MODEL_PATH)
CONVERTED_PATH = "./converted_audio.wav"
GET_DF = True
SPEECH_MODEL_PATH = "emotion_model_29_01_2023.h5"
SPEECH_CLASSIFICATION = tf.keras.models.load_model(SPEECH_MODEL_PATH)
encoder = {0:"anger", 1:"fear", 2:"joy", 3:"love", 4:"sadness", 5:"surprise"}
RESIZE_SIZES = {
    "single":(50, 50),
    "double":(100, 50)
}
RESIZE = True
FACE_PERC = 0.6
N_FPS = 1
SHOW = False
MODE = "rgb"
USE = "both" #either single or both
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier", )

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# CHUNK_SIZE = 30
pipe = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-tiny",
    chunk_length_s=CHUNK_SIZE,
    device=device,
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/submit_video", methods = ["POST"])
def process():
    video = request.files.get("video")
    video.save("./abc.mp4")
    cap = cv2.VideoCapture("./abc.mp4")
    if not cap.isOpened():
        return redirect("/")
    else:
        try:
            eyes_list, both_eyes, no_eyes, no_face, face_emotions = face_eye_emotion(
            video_path="./abc.mp4",
            face_perc= FACE_PERC, n_fps= N_FPS, resize= RESIZE, show=SHOW, mode= MODE, resize_sizes= RESIZE_SIZES, 
            FACE_CASCADE= FACE_CASCADE, EYE_CASCADE= EYE_CASCADE, emotion_model= FACE_EMOTION_MODEL, CLASSES_EMO_FACE = CLASSES_EMO_FACE
             )
            face_emotion_eye = post_process(eyes_list= eyes_list, both_eyes= both_eyes, use=  USE, face_emotion= face_emotions, no_eyes= no_eyes, no_face= no_face)
            convert("./abc.mp4", CONVERTED_PATH)
            text_results = speech2text_pipeline(pipe)
            divide_clips(text_results["timestamp"].values,CONVERTED_PATH)
            sound_class = make_inferences(classifier)
            results = classify_speech(text_results)
            return render_template("analyzed.html", neutral_face = face_emotion_eye["Emotion data"]["Neutral"],
                                                     happy_face = face_emotion_eye["Emotion data"]["Happy"],
                                                     pay_attention = face_emotion_eye["Attention %"],
                                                     neutral = sound_class["neutral"], happy = sound_class["happy"], sad = sound_class["sad"], text_data = results
            )
        # return render_template("analyzed.html", neutral_face = 20,
        #                                               happy_face = 30,
        #                                              pay_attention = 40, neutral = 30, happy = 40, sad = 50, text_data = {"a":1}
        #     )

        except Exception as exc:
            return f"Some error occured while doing calculation due to exception : {type(exc).__name__} : {exc}"
        


if __name__ == "__main__":
    app.run(host="localhost", debug=True)


# line43

            #      <!-- {% for key, value in text_data.items() %} -->
            #     <!-- {{ key }} -->
            #     <!--<p>
            #         <span {% if value=="anger" %} class="green-text" {% endif %} 
            #         {% if value=="fear" %} class="yellow-text" {% endif%} 
            #         {% if value=="joy" %} class="orange-text" {% endif %} 
            #         {% if value=="love" %} class="red-text" {% endif %} 
            #         {% if value=="sadness" %} class="black-text" {% endif %}
            #         {% if value=="surprise" %} class="brown-text" {% endif %}>
            #             {{ value }}
            #         </span> 
            #     </p> -->
            #     <!-- {% endfor %} -->
            # </div>