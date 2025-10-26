import os
os.environ["PATH"] += os.pathsep + r"C:\Users\shahy\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-essentials_build\bin"

from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

print("ffmpeg path:", AudioSegment.converter)
print("ffprobe path:", AudioSegment.ffprobe)
import os
import pandas as pd
import numpy as np
import re
import string
import pickle
import librosa
import shutil
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from collections import Counter
from sklearn.ensemble import IsolationForest
from pydub import AudioSegment
from typing import List, Tuple
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import pandas as pd
import torch
from transformers import pipeline
import wave
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# load model and processor
from speechbrain.pretrained.interfaces import foreign_class
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
stop_words = set(stopwords.words('english'))
global SPEECH_CLASSIFICATION, tokenizer

#stop_words = stopwords.words('english')

def crop(img, x, y, width, height):
    return img[y:y+height, x:x+width]

#CLASSES_EMO_FACE = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


def face_predict(img, model, CLASSES_EMO_FACE):

    img = cv2.resize(img, (48, 48))
    # cv2.imshow("Face", img)
    # cv2.waitKey(0)
    img = img/255
    img = np.expand_dims(img, axis=2)
    # print(img.shape)
    predicts = model.predict(np.expand_dims(img, axis = 0), verbose = 0)
    # post_process(predicts)
    return CLASSES_EMO_FACE[np.argmax(predicts[0])]


#VIDEO_PATH = "merged_new.mp4"
#FACE_CASCADE = cv2.CascadeClassifier("haarcascade_face.xml") 
#FACE_EMOTION_MODEL_PATH = "model_mv1.h5"

#FACE_EMOTION_MODEL = tf.keras.models.load_model(FACE_EMOTION_MODEL_PATH)
# model.load_weights(FACE_EMOTION_MODEL_PATH)
# FACE_EMOTION_MODEL = model
# RESIZE_SIZES = {
#     "single":(50, 50),
#     "double":(100, 50)
# }
# neutral = True
# happy = True
# RESIZE = True
# FACE_PERC = 0.6
# N_FPS = 1
# SHOW = False

#EYE_CASCADE = cv2.CascadeClassifier("haarcascade_eye.xml")  # capture frames from a camera 
# MODE = "rgb"
def face_eye_emotion(video_path:str, face_perc:float, n_fps:int, resize:bool, show:bool, mode:str, resize_sizes:dict, FACE_CASCADE,EYE_CASCADE, emotion_model, CLASSES_EMO_FACE):
    neutral = True
    happy = True
    cap = cv2.VideoCapture(video_path) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        return f"Unable to read given file {video_path}"
    val = int(fps/n_fps)
    eyes_list = []
    both_eyes = []
    no_face = 0
    no_eyes = 0
    face_emotions = {}
    assert mode in  ["gray", "rgb"], f"mode must be either gray or rgb not {mode}"
    # loop runs if capturing has been initialized. 
    while 1:  
        # reads frames from a camera 
        ret, img = cap.read()  
        fno = int(cap.get(1))
        if not ret:
            break   
        # print("here")
        if fno % val != 0:
            continue
        # convert to gray scale of each frames 
        # cv2.imshow('img',img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # print(fno)
        # Detects faces of different sizes in the input image 
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5) 
        if len(faces) < 1:
            # print("no_face")
            no_face += 1
            face_emotions[fno] =None
            continue
        
        # cv2.imshow('img',img)
        for (x,y,w,h) in faces[:1]: 
            # To draw a rectangle in a facee  
            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
            roi_gray = gray[y:y+h, x:x+w] 
            roi_color = img[y:y+h, x:x+w] 
            face_emotions[fno] = face_predict(roi_gray, emotion_model, CLASSES_EMO_FACE)
            if neutral:
                if face_emotions[fno] == "Neutral":
                    neut_image = roi_color
                    cv2.imwrite("static/neutral.jpg", neut_image)
                    neutral = False
            if happy:
                if face_emotions[fno] == "Happy":
                    happ_image = roi_color
                    cv2.imwrite("static/happy.jpg", happ_image)
                    happy = False
            # post_process(predicts)
            
            # cv2.imshow('img',img)
            # Defensive checks before detecting eyes to avoid passing empty/invalid images
            # and to handle cascade load failures.
            if hasattr(EYE_CASCADE, 'empty') and EYE_CASCADE.empty():
                # Eye cascade failed to load (wrong path or missing file). Skip eye detection.
                no_eyes += 1
                continue

            if roi_gray is None or roi_gray.size == 0 or roi_gray.shape[0] < 10 or roi_gray.shape[1] < 10:
                # ROI is too small or invalid for detection
                no_eyes += 1
                continue

            try:
                # Use explicit parameters and a minimum size to be more robust
                eyes = EYE_CASCADE.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
            except cv2.error:
                # In rare cases OpenCV can still raise; skip this frame gracefully
                no_eyes += 1
                continue
            #To draw a rectangle in eyes 
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255,0),1)
            count = 0
            ex_list = []
            ey_list = []
            x_end = []
            y_end = []
            if len(eyes) < 2:
                # eyes_list.append([None, None])
                # both_eyes.append(None)
                # print("no eyes")
                no_eyes += 1
                # eyes_list.append([None, None])
                # both_eyes.append(None)
                continue
            temp_list = []
            # cv2.imshow('img',img)
            for (ex,ey,ew,eh) in eyes:
                if ey + eh <= roi_gray.shape[1] * face_perc:
                    count += 1
                    ex_list.append(ex)
                    ey_list.append(ey)
                    x_end.append(ex + ew)
                    y_end.append(ey + eh)
                    if mode == "gray":
                        temp_list.append(crop(roi_gray, ex, ey, ew, eh))
                        cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
                    else:
                        temp_list.append(crop(roi_color, ex, ey, ew, eh))
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
                
            
        # print(ex_list)
        # print(ey_list)
        # Display an image in a window
        if count >= 2:
            if ex_list[0] < ex_list[1]:
                pass
            else:
                # print('swapped')
                temp_list[1], temp_list[0] = temp_list[0], temp_list[1]
            if resize:
                temp_list[0] = cv2.resize(temp_list[0], resize_sizes["single"])
                temp_list[1] = cv2.resize(temp_list[1], resize_sizes["single"])
            eyes_list.append(temp_list)
            # print(2)
            if show:
                cv2.imshow("eye1",cv2.hconcat([temp_list[0], temp_list[1]]))
            # cv2.imshow("eye2",temp_list[1])
            min_ex = min(ex_list)
            min_ey = min(ey_list)        
            max_ex = max(x_end)
            max_ey = max(y_end)
            # req_w = max_ex - min_ex
            # req_h = max_ey - min_ey
            # cv2.rectangle(roi_gray,(min_ex,min_ey),(max_ex, max_ey),(0,127,255),2)
            if mode == "gray":
                both = crop(roi_gray, min_ex, min_ey, max_ex - min_ex, max_ey - min_ey)   
            else:
                both = crop(roi_color, min_ex, min_ey, max_ex - min_ex, max_ey - min_ey)   
            if resize:
                both = cv2.resize(both, resize_sizes["double"])
            both_eyes.append(both)

            
            if show:
                cv2.imshow("both_eyes", both)
                cv2.imshow('img',img)
                cv2.imshow("new", roi_gray)
        else:
            # eyes_list.append([None, None])
            # both_eyes.append(None)
            continue
        # Wait for Esc key to stop 
        # cv2.imshow('img',img)
        # print(face_emotions)
        # k = cv2.waitKey(5)
        # if k == 27: 
        #     break
        # cv2.waitKey(0)
        # break
        
    
    # Close the window 
    cap.release() 
    
    # De-allocate any associated memory usage 
    cv2.destroyAllWindows() 
    return eyes_list, both_eyes, no_eyes, no_face, face_emotions


def post_process(eyes_list:list, both_eyes:list, use:str, face_emotion:dict, no_eyes, no_face, n_estimators :int = 100,):
    emotions = Counter(face_emotion.values())
    if use == "single":
        eyes_stacked = list(map(lambda x: np.vstack([x[0], x[1]]).flatten(), eyes_list))
    else:
        eyes_stacked = np.array(list(map(lambda x:x.flatten(), both_eyes)))
    iso_model = IsolationForest(n_estimators=n_estimators)
    predicts = iso_model.fit_predict(eyes_stacked)
    contact = np.sum(predicts == 1)
    perc_conc = contact/ (len(predicts) + no_eyes + no_face) * 100
    return {"Emotion data" : emotions, "Attention %" : perc_conc}

    
# converts mp4 to wav

def convert(mp4_path:str, wav_save_path:str):
    assert mp4_path[-3:].lower() == "mp4"

    # load mp4
    audio = AudioSegment.from_file(mp4_path)
    
    # export to dir
    audio.export(wav_save_path, format="wav")

    # return True

def speech2text_pipeline(pipe, return_df = True):
    sample = {}
    # sample
    sample['path'] = "./converted_audio.wav"
    sample['array'], sample["sampling_rate"] = librosa.load("./converted_audio.wav", sr=16000)
    prediction = pipe(sample, return_timestamps=True)["chunks"]
    return pd.DataFrame(prediction) if return_df else prediction


def divide_clips(timestamps:List[Tuple], wav_path: str, temp_file="temp_clips"):
    # Open the wave file
    with wave.open(wav_path, "rb") as wave_file:
        # Get the number of frames and the frame rate
        n_frames = wave_file.getnframes()
        frame_rate = wave_file.getframerate()
        # print(frame_rate)
        # print(frame_rate)
        # # frame_rate = wave_file.getframerate()
        # print(n_frames, frame_rate)
        
        # Create the temp folder if it doesn't exist
        if not os.path.exists(f"{temp_file}"):
            os.mkdir(f"{temp_file}")
        
        # Iterate over the start and end times
        for i, (start, end) in enumerate(timestamps):
            # print(int(start_frame))
            # Calculate the start and end frame indices
            start_frame = int(start * frame_rate)
            end_frame = int(end * frame_rate)
            
            # print(start, end, start_frame, end_frame)
            
            # Skip the iteration if the start frame is greater than the end frame
            if start_frame >= end_frame:
                continue
            # print(int(start_frame))
            # print(wave_file.getnframes())
            # print(start_frame * frame_rate)
            # Read the audio data for the specified frame range
            try:
                wave_file.setpos(start_frame)
            except:
                continue
            audio_data = wave_file.readframes(end_frame - start_frame)
            
            # Save the audio data to a new wave file
            with wave.open(f"{temp_file}/part{i}.wav", "wb") as output_file:
                output_file.setnchannels(wave_file.getnchannels())
                output_file.setsampwidth(wave_file.getsampwidth())
                output_file.setframerate(frame_rate)
                output_file.writeframes(audio_data)

# load model and make predictions over all the clips


def make_inferences(classifier, remove_dir = True):
    # loading model
    # make inferences over all the clips
    clips_dir = "./temp_clips/"
    clips = [os.path.join(clips_dir, clip_name) for clip_name in os.listdir(clips_dir)]
    text_labs = []
    for idx, clip in enumerate(clips):
        ti = time.time()
        try:
            out_prob, score, index, text_lab = classifier.classify_file(clip)
        except:
            pass
        text_labs.append(text_lab[0])
        # print(f"completed part{idx} in {time.time()-ti}", score.numpy()[0], text_lab[0])
    shutil.rmtree(clips_dir)
    return text_labs

def get_res(key):
    if key.lower() == "hap":
        return "happy"
    elif key.lower() == "neu":
        return "neutral"
    elif key.lower() == "ang":
        return 'sadness'

def lower_text(text):
    return text.lower()

def remove_number(text):
    num = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
    return num.sub(r'', text)

def remove_punct(text):
    punctuations = string.punctuation 
    
    for p in punctuations:
        text = text.replace(p, '')
    return text
    
def remove_stopwords(text):
    text = ' '.join([word for word in text.split() if word not in (stop_words)])
    return text

def clean_text(text):
    text = lower_text(text)
    text = remove_number(text)
    text = remove_punct(text)
    text = remove_stopwords(text)
    
    return text


def emotion_predictor(cleaned_text, SPEECH_CLASSIFICATION, tokenizer):
    # print(cleaned_text)
    encoder = {0:"anger", 1:"fear", 2:"joy", 3:"love", 4:"sadness", 5:"surprise"}

    text_to_seq = tokenizer.texts_to_sequences([cleaned_text])    
    # print(text_to_seq)

    text_to_seq = pad_sequences(text_to_seq, padding='post', maxlen=35)
    # print(text_to_seq)

    return encoder.get(np.argmax(SPEECH_CLASSIFICATION.predict(text_to_seq, verbose = 0), axis = 1)[0])

def classify_speech(df, SPEECH_CLASSIFICATION, tokenizer):
    res = []
    for i in df["text"]:
        cleaned = clean_text(i)
        res.append({i: emotion_predictor(cleaned, SPEECH_CLASSIFICATION, tokenizer)})
    return res

def speech2text_pipeline(pipe, audio_path): # Argument name must match the call in app.py
    """
    Performs speech-to-text using the provided pipeline and audio file path.
    Args:
        pipe: The Hugging Face ASR pipeline.
        audio_path (str): The full path to the WAV audio file.
    Returns:
        pandas.DataFrame or None: DataFrame with 'timestamp' and 'text', or None on error.
    """
    try:
        print(f"Loading audio for ASR from: {audio_path}") # Log the path being used
        # --- *** USE THE PASSED-IN audio_path *** ---
        sample = {}
        # Make sure librosa can handle the path format (e.g., backslashes on Windows)
        sample['array'], sample["sampling_rate"] = librosa.load(audio_path, sr=16000) # Use the argument here
        # ---

        print("Running ASR pipeline...")
        # Make sure the 'pipe' function expects the sample format correctly
        output = pipe(sample.copy(), batch_size=8, return_timestamps=True)
        print("ASR pipeline finished.")

        if output and 'chunks' in output:
            df = pd.DataFrame(output['chunks'])
            # Ensure columns are correctly named based on pipe output
            if not df.empty and len(df.columns) >= 2:
                 # Assuming the first col is timestamp, second is text
                 df.columns = ["timestamp", "text"] + list(df.columns[2:]) # Keep extra cols if any
                 # Check timestamp format - Whisper might return dicts like {'timestamp': (0.0, 2.5)}
                 if not df['timestamp'].empty and isinstance(df['timestamp'].iloc[0], tuple):
                      print("Timestamps appear valid.")
                 else:
                      print("Warning: Timestamps might not be in the expected tuple format.")
                 return df
            else:
                 print("ASR output 'chunks' DataFrame is empty or has unexpected columns.")
                 return None # Return None or an empty DataFrame as appropriate
        else:
            print("ASR output format unexpected or empty. Output:", output)
            return None

    except FileNotFoundError:
        print(f"Error in speech2text_pipeline: Audio file not found at {audio_path}")
        return None
    except Exception as e:
        print(f"Error during speech-to-text (librosa load or pipeline): {e}")
        traceback.print_exc() # Print full traceback for debugging
        return None


# --- Keep all your other functions (face_eye_emotion, post_process, convert, etc.) the same ---
# ... (rest of your functions.py code) ...
# Example placeholder for classify_speech if needed for context
def classify_speech(text_results_df, model, tokenizer):
    # Dummy implementation - replace with your actual logic
    # Assumes text_results_df is the DataFrame from speech2text_pipeline
    results_list = []
    if text_results_df is not None and 'text' in text_results_df.columns:
        for text in text_results_df['text']:
            # Replace with your actual model prediction logic
            predicted_emotion = "neutral" # Placeholder
            results_list.append({text: predicted_emotion})
    return results_list