from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

data = pd.read_csv('medical_data.csv')

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Patient_Problem'])

label_encoder_disease = LabelEncoder()
disease_labels = label_encoder_disease.fit_transform(data['Disease'])
label_encoder_prescription = LabelEncoder()
prescription_labels = label_encoder_prescription.fit_transform(data['Prescription'])

model = load_model('model.h5')

max_length = 17 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    patient_problem = request.form['patient_problem']
    sequence = tokenizer.texts_to_sequences([patient_problem])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence)
    
    disease_index = np.argmax(prediction[0])
    prescription_index = np.argmax(prediction[1])
    
    disease_predicted = label_encoder_disease.inverse_transform([disease_index])[0]
    prescription_predicted = label_encoder_prescription.inverse_transform([prescription_index])[0]
    
    return render_template('result.html', patient_problem=patient_problem, disease=disease_predicted, prescription=prescription_predicted)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))


# import streamlit as st
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder

# # Load data and model
# data = pd.read_csv('medical_data.csv')
# model = load_model('model.h5')

# # Setup tokenizer
# tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
# tokenizer.fit_on_texts(data['Patient_Problem'])

# # Setup label encoders
# label_encoder_disease = LabelEncoder()
# label_encoder_disease.fit(data['Disease'])

# label_encoder_prescription = LabelEncoder()
# label_encoder_prescription.fit(data['Prescription'])

# max_length = 17

# # Streamlit UI
# st.set_page_config(page_title="Medical Diagnoser", layout="centered")
# st.title("🩺 AI Medical Diagnoser")

# st.markdown("Enter a description of your symptoms:")

# patient_problem = st.text_input("Patient Problem", placeholder="e.g. severe headache and nausea")

# if st.button("Predict"):
#     if patient_problem.strip() == "":
#         st.warning("Please enter your symptoms to get a prediction.")
#     else:
#         sequence = tokenizer.texts_to_sequences([patient_problem])
#         padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
#         prediction = model.predict(padded_sequence)

#         disease_index = np.argmax(prediction[0])
#         prescription_index = np.argmax(prediction[1])

#         disease_predicted = label_encoder_disease.inverse_transform([disease_index])[0]
#         prescription_predicted = label_encoder_prescription.inverse_transform([prescription_index])[0]

#         st.success(f"### 🧬 Predicted Disease: {disease_predicted}")
#         st.info(f"💊 Recommended Prescription: {prescription_predicted}")