import sounddevice as sd
from scipy.io.wavfile import write
import keyboard
import time
import whisper
from whisper.audio import log_mel_spectrogram, pad_or_trim
import joblib
import os
import re

# Setupt Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
AUDIO_PATH = os.path.join(BASE_DIR, "output.wav")

# Load Model

model = whisper.load_model("small")
sentiment_model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Creating Methods

def record_audio():

    print("Press enter to start record")
    keyboard.wait("enter")
    print("recording....., press enter again to stop")

    start_time = time.time()

    fs = 16000
    channels = 1
    duration = 10

    # start recording with an arbitary large buffer
    recording = sd.rec(int(fs*duration),samplerate=fs,channels=channels)

    keyboard.wait("enter")
    print("stoping")
    print()

    sd.stop()

    # calculate actual duration 
    duration = time.time() - start_time

    # svaing only the recorded portion
    write("output.wav",fs,recording[:int(duration*fs)])


def speech_to_text():
    
    global model
    #audio_file = open("output.wav","rb")
    audio = whisper.load_audio("output.wav")
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio,n_mels=model.dims.n_mels)

    options = whisper.DecodingOptions()
    decoding_result = whisper.decode(model,mel,options)
    result = decoding_result.text

    # Cleaning the Whisper text
    text = result.lower()
    text = re.sub(r"\b(uh|um|you know|actually|basically)\b", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def predict_sentiment():

    while True:
        record_audio()

        output = speech_to_text()
        print()
        print("Customer statement: ",output)

        text_vector = vectorizer.transform([output])
        prediction = sentiment_model.predict(text_vector)[0]
        #prediction_label = vectorizer.inverse_transform([prediction])[0]
        
        print()
        print("Sentiment:")
        if prediction == 0:
            print("Negative Review")
        elif prediction == 1:
            print("Neutral review")
        else:
            print("Positive Review")

        print()

if __name__ == "__main__":
    predict_sentiment()

