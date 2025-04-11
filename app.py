import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf
import tempfile
import matplotlib.pyplot as plt
from matplotlib import colormaps
import tempfile as tp

# --------------------------------- PARTE 1: EXTRAIR FEATURES --------------------------------- #

# Carregar o modelo e o scaler
MODEL_PATH = r"C:\Users\slash\OneDrive\Área de Trabalho\TRILHA\Mini_projeto_2\miniprojeto2\models\audio_emotion_model.keras"  # Example
SCALER_PATH = r"C:\Users\slash\OneDrive\Área de Trabalho\TRILHA\Mini_projeto_2\miniprojeto2\models\scaler.jolib"              # Example

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de emoções
EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]


# Função para extrair features
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, sr=16000, mono=True)
    features = []

    # Zero Crossing Rate
    # Extract the zcr here
    # features.extend(zcr)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y = data), axis=1)

    features = np.hstack((features, zcr))

    # Chroma STFT
    # Extract the chroma stft here
    # features.extend(chroma)

    chroma = np.mean(librosa.feature.chroma_stft(y =data, sr= sr), axis=1)

    features = np.hstack((features, chroma))

    # MFCCs
    # Extract the mfccs here
    # features.extend(mfccs)

    mfccs = np.mean(librosa.feature.mfcc(y= data, sr= sr), axis=1)

    features = np.hstack((features, mfccs))

    # RMS
    # Extract the rms here
    # features.extend(rms)

    rms = np.mean(librosa.feature.rms(y =data), axis=1)

    features = np.hstack((features, rms))

    # Mel Spectrogram
    # Extract the mel here
    # features.extend(mel)

    mel = np.mean(librosa.feature.melspectrogram(y =data, sr= sr), axis=1)

    features = np.hstack((features, mel))

    # Garantir que tenha exatamente 162 features (ou truncar/zerar)
    target_length = 162
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]

    return np.array(features).reshape(1, -1)


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #


# Configuração do app Streamlit (Título e descrição)
# Code here

st.title('Emotions in audios!')
st.write('Selecione um áudio para analisarmos!')    

# Upload de arquivo de áudio (wav, mp3, ogg)
uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    
    # Salvar temporariamente o áudio
    # Code here

    temp_file = tp.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(uploaded_file.getvalue())
    audio_path = temp_file.name
    temp_file.close()

    # Reproduzir o áudio enviado
    # Code here

    st.audio(uploaded_file)

    # Extrair features
    # Code here

    features = extract_features(audio_path)

    # Normalizar os dados com o scaler treinado
    # Code here

    features_scaled = scaler.transform(features)

    # Ajustar formato para o modelo
    # Code here

    features_final = np.expand_dims(features_scaled, axis=2)

    # Fazer a predição
    # Code here

    predictions = model.predict(features_final)
    emotion = EMOTIONS[np.argmax(predictions[0])]

    # Exibir o resultado
    # Code here

    st.success(f"Analise feita emoção: {emotion}")

    # Exibir probabilidades (gráfico de barras)
    # Code here

    colors = ['#a0b4fc']
    classes = EMOTIONS
    fig, ax = plt.subplots()
    ax.set_ylabel("Probabilidade")
    ax.bar(classes, predictions[0],color = colors)
    st.pyplot(fig)

    st.write("Probabilidades:")
    for emotion, prob in zip(EMOTIONS, predictions[0]):
        st.write(f"{emotion}: {prob*100:.1f}%")

    # Remover o arquivo temporário
    # Code here

    os.remove(audio_path)

