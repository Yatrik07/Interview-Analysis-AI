# Online Interview Analysis AI

An AI-powered web application to analyze interview performance from video recordings, providing data-driven feedback on language, tone, and non-verbal cues.

## Problem Solved

The job interview is a critical step, yet candidates often lack objective feedback on their performance. This tool provides a comprehensive evaluation by analyzing mock interview videos using deep learning. It extracts insights on language, vocal tone, facial expressions, and eye movement, generating a report highlighting strengths and areas for improvement. This empowers individuals to refine their skills and increase their chances of success.

## Features

**Speech & Language Analysis**  
- Automatic Speech Recognition (ASR) to transcribe the interview.  
- Pitch modulation analysis to gauge vocal confidence and variation.  
- Natural Language Processing (NLP) to analyze sentiment and emotion in the transcript.  

**Vocal Tone Analysis**  
- Emotion recognition directly from the audio signal.  

**Facial Emotion Recognition**  
- Detects emotions (Neutral, Happy, Sad, Angry, etc.) from facial expressions frame-by-frame.  

**Eye Tracking & Attention**  
- Monitors eye movement using computer vision to estimate the percentage of time the candidate maintains eye contact (attention).  
- Includes anomaly detection for blinking patterns.  

**Comprehensive Reporting**  
- Integrates all analyses into a user-friendly report.  

## Technology Stack

- **Backend:** Flask (Python Web Framework)  
- **Machine Learning / Deep Learning:**  
  - TensorFlow / Keras (CNN for Facial Emotion, CNN-BiLSTM for Text Emotion)  
  - Transformers (Hugging Face - Whisper for ASR)  
  - SpeechBrain (Vocal Emotion Recognition)  
  - Scikit-learn (Isolation Forest for Anomaly Detection)  
- **Audio Processing:** pydub, librosa, wave  
- **Computer Vision:** OpenCV (Haar Cascades for Face/Eye Detection)  
- **NLP:** NLTK (Stopwords)  
- **Frontend:** HTML, CSS, JavaScript (including Chart.js for visualization)  
- **Deployment:** Docker  

## How It Works

1. **Upload:** The user uploads an interview video via the web interface.  
2. **Processing (Backend):**  
   - Flask backend saves the video.  
   - **Video Analysis:** OpenCV detects faces, eyes, and facial emotions. Eye tracking data is analyzed using Isolation Forest.  
   - **Audio Extraction:** pydub (using FFmpeg) extracts the audio track.  
   - **Transcription:** Whisper ASR transcribes the audio, generating text and timestamps.  
   - **Audio Segmentation:** Audio is split into smaller clips based on ASR timestamps.  
   - **Vocal Emotion:** SpeechBrain analyzes each audio clip for vocal tone emotion.  
   - **Text Emotion:** Text is cleaned, tokenized, and analyzed using the CNN-BiLSTM model.  
   - **Report Generation:** Results from all analyses are aggregated.  
   - **Display:** Flask renders `analyzed.html`, showing results with charts and tables.  

## Prerequisites

- Python 3.8+  
- FFmpeg (essential for audio processing, ensure it's added to your system's PATH)  
- (Optional) NVIDIA GPU with CUDA drivers for faster model inference  

## Installation & Setup

## Download Pre-trained Models & Assets

- Haar Cascade XML files: `haarcascade_face.xml`, `haarcascade_eye.xml`  
- Trained model files: `model_mv1.h5`, `emotion_model_29_01_2023.h5`  
- Tokenizer file: `tokenizer.pkl`  
- Hugging Face and SpeechBrain models may download automatically on first run if not cached  

### Clone the repository
```bash
git clone [YOUR_GITHUB_REPO_LINK_HERE]
cd [YOUR_PROJECT_DIRECTORY_NAME]
```

### Virtual Environment
```Python
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```
### Install Dependencies
```bash
pip install -r requirements.txt
# Download NLTK stopwords
python -m nltk.downloader stopwords
```

## Usage

1. Navigate to [http://localhost:5000](http://localhost:5000)  
2. Upload your interview video  
3. Click "Analyze Video"  
4. Wait for processing to complete  
5. View the generated report with insights on facial emotion, attention, vocal tone, and text emotion  

## Challenges Faced

- Accuracy of facial and text emotion recognition  
- Detecting eye blinking patterns and anomalies  
- Addressed by fine-tuning models, using diverse datasets, and applying Isolation Forest for anomaly detection  

## Open Source Contribution

This project uses open-source libraries: TensorFlow, Keras, Transformers, SpeechBrain, OpenCV, NLTK, Flask.  
The code is publicly available on GitHub at [YOUR_GITHUB_REPO_LINK_HERE]. Contributions, feedback, and bug reports are welcome.  

## Developer

**Yatrikkumar Shah**  
[LinkedIn](https://www.linkedin.com/in/yatrikkumar-shah/)  

## License

This project is licensed under the MIT License. See the `LICENSE.md` file for details.


