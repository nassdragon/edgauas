import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load models
with open('supervised/perceptron/model_pcp_fish.pkl', 'rb') as file:
    fish_model_pcp = pickle.load(file)
with open('supervised/perceptron/model_pcp_fruit.pkl', 'rb') as file:
    fruit_model_pcp = pickle.load(file)
with open('supervised/perceptron/model_pcp_pumpkin.pkl', 'rb') as file:
    pumpkin_model_pcp = pickle.load(file)

# Load encoders
with open('supervised/perceptron/label_encoder_fish.pkl', 'rb') as file:
    fish_encoder = pickle.load(file)
with open('supervised/perceptron/label_encoder_fruit.pkl', 'rb') as file:
    fruit_encoder = pickle.load(file)
with open('supervised/perceptron/label_encoder_pumpkin.pkl', 'rb') as file:
    pumpkin_encoder = pickle.load(file)

# Example for loading the XGBoost model for fish
with open('supervised/xgboost/model_xgb_fish.pkl', 'rb') as file:
    fish_model_xgboost = pickle.load(file)

with open('supervised/xgboost/label_encoder_xgbfish.pkl', 'rb') as file:
    fish_encoder_xgboost = pickle.load(file)

# Example for loading the XGBoost model for fruit
with open('supervised/xgboost/model_xgb_fruit.pkl', 'rb') as file:
    fruit_model_xgboost = pickle.load(file)

with open('supervised/xgboost/label_encoder_xgb_fruit.pkl', 'rb') as file:
    fruit_encoder_xgboost = pickle.load(file)

# Title and description
st.title('Prediksi Jenis Ikan atau Buah menggunakan Machine Learning')
st.write("## Tentukan jenis ikan atau buah ")

# Choose category
st.write("### Pilih Kategori")
option = st.selectbox("Klasifikasi:", ("Fish", "Fruit", "Wine"))

# Choose algorithm
st.write("### Pilih Algoritma untuk Prediksi")
if option == "Wine":
    algorithm_option = st.selectbox("Pilih Algoritma:", ("KMeans", "Perceptron", "XGboost"))
else:
    algorithm_option = st.selectbox("Pilih Algoritma:", ("Perceptron", "XGboost"))

st.markdown("---")

# Input data for Fish and Fruit
with st.form(key='my_form'):
    if option == "Fish":
        st.write("### Masukkan Data Ikan")
        weight = st.number_input('Berat Ikan (dalam gram)', min_value=0.0, format="%.2f")
        length = st.number_input('Panjang Ikan (dalam cm)', min_value=0.0, format="%.2f")
        height = st.number_input('w_l_Ratio', min_value=0.0, format="%.2f")
        
        submit = st.form_submit_button(label='Prediksi Jenis Ikan', help="Klik untuk melihat hasil prediksi")
        
        if submit:
            input_data = np.array([weight, length, height]).reshape(1, -1)
            if algorithm_option == "Perceptron":
                prediction = fish_model_pcp.predict(input_data)
                # Decode hasil prediksi
                fish_result = fish_encoder.inverse_transform(prediction)[0]
            elif algorithm_option == "XGboost":
                prediction = fish_model_xgboost.predict(input_data)
                # Decode hasil prediksi
                fish_result = fish_encoder_xgboost.inverse_transform(prediction)[0]
                
            st.success(f"### Jenis Ikan: {fish_result}")

    elif option == "Fruit":
        st.write("### Masukkan Data Buah")
        weight = st.number_input('Berat Buah (dalam gram)', min_value=0.0, format="%.2f")
        diameter = st.number_input('Diameter Buah', min_value=0.0, format="%.2f")
        red = st.number_input('Skor Warna Buah Merah', 0, 255, 0)
        green = st.number_input('Skor Warna Buah Hijau', 0, 255, 0)
        blue = st.number_input('Skor Warna Buah Biru', 0, 255, 0)
        
        submit = st.form_submit_button(label='Prediksi Jenis Buah', help="Klik untuk melihat hasil prediksi")
        
        if submit:
            input_data = np.array([diameter, weight, red, green, blue]).reshape(1, -1)
            if algorithm_option == "Perceptron":
                prediction = fruit_model_pcp.predict(input_data)
                # Decode hasil prediksi
                fruit_result = fruit_encoder.inverse_transform(prediction)[0]
            elif algorithm_option == "XGboost":
                prediction = fruit_model_xgboost.predict(input_data)
                # Decode hasil prediksi
                fruit_result = fruit_encoder_xgboost.inverse_transform(prediction)[0]
    
            st.success(f"### Jenis Buah: {fruit_result}")

# Wine classification (KMeans clustering)
if option == "Wine":
    st.write("### Klasifikasi Data Wine")
    if algorithm_option == "KMeans":
        # Input for maximum K value
        max_k = st.slider("Pilih Maksimal K", min_value=2, max_value=20, value=10)

        # Add slider for selecting K (number of clusters)
        k_value = st.slider("Pilih Jumlah Kluster K", min_value=2, max_value=max_k, value=3)

        # Add submit button for KMeans classification
        submit_wine = st.button("Tampilkan Hasil KMeans")

        if submit_wine:
            try:
                # Load KMeans model for Wine
                with open("unsupervised/kmean_wine.pkl", "rb") as file:
                    kmeans_model = pickle.load(file)

                # Simulate wine dataset
                wine_data = pd.DataFrame({
                    "alcohol": np.random.uniform(10, 15, 200),
                    "total_phenols": np.random.uniform(0.1, 5, 200),
                })
                wine_features = wine_data.values

                # Calculate SSE for each K value (Elbow Method)
                sse = []
                for k in range(1, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(wine_features)
                    sse.append(kmeans.inertia_)

                # Plot Elbow Method graph
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, max_k + 1), sse, marker='o')
                plt.xlabel("Jumlah Kluster (K)")
                plt.ylabel("Sum of Squared Errors (SSE)")
                plt.title("Elbow Method untuk Menentukan Nilai Optimal K")
                plt.grid(True)

                # Display the plot in Streamlit
                st.pyplot(plt)

                # Perform KMeans with selected K
                kmeans = KMeans(n_clusters=k_value, random_state=42)
                wine_data['cluster'] = kmeans.fit_predict(wine_features)

                # Display the cluster assignments
                st.write("### Hasil Klasterisasi Wine")
                st.write(wine_data.head())

            except FileNotFoundError:
                st.error("Model kmean_wine.pkl tidak ditemukan! Pastikan file ada di direktori yang sama.")
    else:
        st.warning("Pilih Algoritma KMeans untuk klasifikasi Wine.")

# Style - Custom CSS Style
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
    }
    .stSlider>.stSliderHeader {
        color: #FF6347;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
