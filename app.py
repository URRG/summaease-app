import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pydub import AudioSegment
import speech_recognition as sr
from PyPDF2 import PdfReader

# Initialize T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def summarize_text(input_text, min_length, max_length):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, min_length=min_length, max_length=max_length)
    summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summarized_text

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Streamlit UI
st.title("Text Summarization App")

input_text = st.text_area("Input Text", "")
upload_file = st.file_uploader("Upload PDF File")
upload_audio = st.file_uploader("Upload Audio File (.mp3)")

min_length = st.number_input("Minimum Summary Length", value=50)
max_length = st.number_input("Maximum Summary Length", value=100)

if st.button("Summarize"):
    if input_text:
        summarized_text = summarize_text(input_text, min_length, max_length)
        st.write("Summarized Text:")
        st.write(summarized_text)
    elif upload_file is not None:
        pdf_text = read_pdf(upload_file)
        summarized_text = summarize_text(pdf_text, min_length, max_length)
        st.write("Summarized Text:")
        st.write(summarized_text)
    elif upload_audio is not None:
        audio_file = AudioSegment.from_file(upload_audio)
        audio_file.export("input_audio.wav", format="wav")

        r = sr.Recognizer()
        with sr.AudioFile("input_audio.wav") as source:
            audio_text = r.listen(source)
            try:
                text = r.recognize_google(audio_text)
                summarized_text = summarize_text(text, min_length, max_length)
                st.write("Summarized Text:")
                st.write(summarized_text)
            except sr.UnknownValueError:
                st.write("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                st.write("Could not request results from Google Speech Recognition service; {0}".format(e))
    else:
        st.write("Please input text, upload a PDF file, or upload an audio file.")
