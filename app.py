import streamlit as st
from helper import decode_sequence,load_vectorization_json,stop_recording,preprocess_audio,text_to_speech,start_recording,load_transformer_model
from helper2 import load_transformer_speech_model
import tensorflow as tf

from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv1D,LayerNormalization,Embedding,Dense
eng_to_spa_transformer=load_transformer_model("eng_to_spa_model.keras")
eng_speech_recognizer_model=load_transformer_speech_model("eng_speech_recognition.keras")
spa_to_eng_transformer=load_transformer_model("spa_to_eng_model.keras")
spa_speech_recognizer_model=load_transformer_speech_model("spa_speech_recognition.keras")

spa_vectorization_config=load_vectorization_json("spa_vectorization_config.json")
spa_vectorization_config1=load_vectorization_json("spa_vectorization_config1.json")
spa_vocab=load_vectorization_json("spa_vocab.json")
spa_vocab1=load_vectorization_json("spa_vocab1.json")
eng_vectorization_config=load_vectorization_json("eng_vectorization_config.json")
eng_vectorization_config1=load_vectorization_json("eng_vectorization_config1.json")
eng_vocab=load_vectorization_json("eng_vocab.json")
eng_vocab1=load_vectorization_json("eng_vocab1.json")
eng_speech_idx_to_char=load_vectorization_json("eng_speech_id_to_chaar.json")
spa_speech_idx_to_char=load_vectorization_json("spa_speech_id_to_char.json")

# Recreate the TextVectorization layers
eng_to_spa_Spa_vectorization = TextVectorization.from_config(spa_vectorization_config)
eng_to_spa_Eng_vectorization = TextVectorization.from_config(eng_vectorization_config)
# Set the vocabularies back to the vectorization layers
eng_to_spa_Spa_vectorization.set_vocabulary(spa_vocab)
eng_to_spa_Eng_vectorization.set_vocabulary(eng_vocab)

spa_to_eng_Eng_vectorization= TextVectorization.from_config(eng_vectorization_config1)
spa_to_eng_Spa_vectorization= TextVectorization.from_config(spa_vectorization_config1)

spa_to_eng_Eng_vectorization.set_vocabulary(eng_vocab1)
spa_to_eng_Spa_vectorization.set_vocabulary(spa_vocab1)
# Parameters
spa_vocab = eng_to_spa_Spa_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))

eng_vocab=spa_to_eng_Eng_vectorization.get_vocabulary()
eng_index_lookup=dict(zip(range(len(eng_vocab)), eng_vocab))

# Streamlit UI
st.title("English Voice Recorder")

if st.button("Start Recording"):
    start_recording()
    st.write("English Recording started...")

if st.button("Stop Recording"):
    audio_file = stop_recording()
    st.write("English Recording stopped.")
    preprocessed_audio=preprocess_audio(audio_file)
    preprocessed_audio = tf.expand_dims(preprocessed_audio, axis=0)  
    target_start_token_idx = 2 
    eng_predictions = eng_speech_recognizer_model.generate(preprocessed_audio, target_start_token_idx)
    eng_decoded_predictions = eng_predictions.numpy()
    eng_predicted_text = ''.join([eng_speech_idx_to_char[idx] for idx in eng_decoded_predictions[0]])
    st.write(eng_predicted_text)
    translated_into_spanish=decode_sequence(eng_to_spa_Eng_vectorization,eng_to_spa_Spa_vectorization,eng_predicted_text,20,spa_index_lookup,eng_to_spa_transformer)
    st.write(translated_into_spanish)
    text_to_speech(translated_into_spanish)

st.title("Spanish Voice Recorder")
if st.button("Start Recording"):
    start_recording()
    st.write("Spanish Recording started...")

if st.button("Stop Recording"):
    audio_file=stop_recording()
    st.write("Spanish Recording stopped.")
    preprocessed_audio=preprocess_audio(audio_file)
    preprocessed_audio=tf.expand_dims(preprocessed_audio,axis=0)
    target_start_token_idx=2
    spa_predictions=spa_speech_recognizer_model.generate(preprocessed_audio,target_start_token_idx)
    spa_decoded_predictions=spa_predictions.numpy()
    spa_predicted_text=''.join([spa_speech_idx_to_char[idx] for idx in spa_decoded_predictions[0]])
    st.write(spa_predicted_text)
    translated_into_english=decode_sequence(spa_to_eng_Spa_vectorization,spa_to_eng_Eng_vectorization,spa_predicted_text,20,eng_index_lookup,spa_to_eng_transformer)
    st.write(translated_into_english)
    text_to_speech(translated_into_english)

st.write("Press 'Start Recording' to begin and 'Stop Recording' to finish.")




