import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

# ELM class
class ELM(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_units=100):
        self.hidden_units = hidden_units

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        # Initialize input weights and biases randomly
        self.input_weights = np.random.randn(X.shape[1], self.hidden_units)
        self.biases = np.random.randn(self.hidden_units)
        
        # Compute hidden layer output using sigmoid activation function
        H = self.sigmoid(np.dot(X, self.input_weights) + self.biases)
        
        # Compute output weights using Moore-Penrose generalized inverse
        self.output_weights = np.dot(np.linalg.pinv(H), y)
        
        return self

    def predict(self, X):
        # Compute hidden layer output using sigmoid activation function
        H = self.sigmoid(np.dot(X, self.input_weights) + self.biases)
        
        # Compute predicted values
        y_pred = np.dot(H, self.output_weights)
        
        # If y_pred has only one dimension, reshape it to have two dimensions
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        
        return y_pred.argmax(axis=1)

# Load data
data = pd.read_csv('data_gizi.csv', encoding='latin1')
df = pd.read_csv('output_data.csv', encoding='latin1')

# Penjelasan tentang data
data_explanation = """
Data ini berisi informasi tentang status gizi anak berdasarkan berat badan terhadap tinggi badan (BB/TB).
"""

# Preprocessing steps
X = df.drop(columns=['BB/TB'])
y = df['BB/TB']
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Normalization
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.3, random_state=42)

# Build and train the ELM model
model = ELM(hidden_units=100)
model.fit(X_train, y_train)

# Evaluate the model using cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(model, X_train, y_train, cv=skf)

# Calculate MAPE and MSE
mape = mean_absolute_percentage_error(y_train, y_pred_cv)
mse = mean_squared_error(y_train, y_pred_cv)

# Streamlit application
def main():
    st.markdown("""
        <style>
            .scrollable { max-height: 400px; overflow-y: auto; }
        </style>
    """, unsafe_allow_html=True)
    # Sidebar / Navigation bar
    st.sidebar.title("Klasifikasi Gizi Balita Dengan Metode ELM")
    page = st.sidebar.radio("Extreme Learning Machine", ["Data", "Preprocessing", "Modeling", "Implementasi"])

    # Main content based on selected page
    if page == "Data":
        st.title("Tentang Data")
        st.header("Data Awal")
        st.write(data_explanation)
        st.write(data.head())
        st.header("Data yang Dipakai")
        st.write(df.head())

    elif page == "Preprocessing":
        st.title("Preprocessing")
        st.header("Data Awal")
        st.write(data_explanation)
        st.write(df.head())

        missing_values = df.isnull().sum()

        st.markdown("""
            <div class="split-container">
                <div style="flex: 1; margin-right: 10px;">
                    <h3>Feature Selection</h3>
                    <p>Kita akan menghapus fitur-fitur yang tidak dipakai dalam melakukan pelatihan.</p>
                    <div class="scrollable">""" + df.to_html(index=False) + """</div>
                </div>
                <div style="flex: 1;">
                    <h3>Pengecekan Missing Values</h3>
                    <p>Sebelum melakukan memproses sebuah data kita harus melakukan reprossecing untuk melakukan pengecekan apakah ada missing value.</p>
                    <div class="scrollable">""" + missing_values.to_frame().to_html(header=False) + """</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.header("Normalisasi Data")
        st.write("Normalisasi dilakukan menggunakan Min-Max Scaler untuk mengubah nilai-nilai pada setiap fitur ke dalam rentang [0, 1].")
        st.subheader("Data Setelah Normalisasi")
        st.write(X_normalized)
        st.write(f"Jumlah baris: {X_normalized.shape[0]}, Jumlah kolom: {X_normalized.shape[1]}")
        
        st.header("Split Data")
        st.write("Data dibagi menjadi data Training dan data Testing dengan perbandingan 70:30.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Training")
            st.write(X_train)
            st.write(f"Jumlah baris: {X_train.shape[0]}, Jumlah kolom: {X_train.shape[1]}")
        with col2:
            st.subheader("Data Testing")
            st.write(X_test)
            st.write(f"Jumlah baris: {X_test.shape[0]}, Jumlah kolom: {X_test.shape[1]}")

    elif page == "Modeling":
        st.title("Model yang Digunakan: ELM")
        
        st.header("Langkah-Langkah Modeling")
        
        # Step 1: Inisialisasi Model
        st.subheader("1. Inisialisasi Model")
        st.write("Model ELM dibangun dengan menyertakan parameter jumlah unit pada hidden layer.")
        st.write(f"Jumlah unit pada hidden layer: {model.hidden_units}")
        
        # Step 2: Training Model
        st.subheader("2. Training Model")
        st.write("Model dilatih menggunakan data training yang sudah dinormalisasi.")
        st.write("Input weights dan biases diinisialisasi secara acak.")
        st.write("Output weights dihitung menggunakan Moore-Penrose generalized inverse dari output hidden layer.")
        
        # Compute hidden layer output
        H = model.sigmoid(np.dot(X_train, model.input_weights) + model.biases)
        
        # Step 3: Evaluasi Menggunakan Cross-Validation
        st.subheader("3. Evaluasi Menggunakan MAPE dan MSE")
        st.write("Model dievaluasi menggunakan metode cross-validation dengan 5-fold.")
        st.write("Hasil prediksi dari cross-validation digunakan untuk menghitung MAPE dan MSE sebagai metrik evaluasi.")
        
        # Show initial weights and biases
        st.subheader("Nilai Bobot Awal")
        st.write("Input weights:")
        st.write(model.input_weights)
        st.write("Biases:")
        st.write(model.biases)
        
        # Compute hidden layer output
        st.subheader("Output Hidden Layer")
        st.write(H)
        
        # Compute sigmoid activation
        st.subheader("Aktivasi Sigmoid Biner")
        st.write(model.sigmoid(np.dot(X_train, model.input_weights) + model.biases))
        
        # Compute Moore-Penrose Generalized Inverse
        st.subheader("Moore-Penrose Generalized Inverse")
        moore_penrose_inverse = np.linalg.pinv(H)
        st.write(moore_penrose_inverse)
        
        # Compute output weights
        st.subheader("Nilai Output Weight")
        output_weights = np.dot(moore_penrose_inverse, y_train)
        st.write(output_weights)
        
        # Compute output layer
        st.subheader("Nilai Output Layer")
        output_layer = np.dot(H, output_weights)
        st.write(output_layer)

        # Show MAPE and MSE
        st.subheader("Metrik Regresi (MAPE & MSE)")
        st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")

    elif page == "Implementasi":
        st.title("Implementasi")
        st.header("Input Data")
        
        berat = st.number_input("Berat (kg)", min_value=0.0, value=0.0)
        tinggi = st.number_input("Tinggi (cm)", min_value=0.0, value=0.0)
        usia = st.number_input("Usia (bulan)", min_value=0.0, value=0.0)
        
        if st.button("Prediksi"):
            if usia == 0 or berat == 0 or tinggi == 0:
                st.warning("Masukkan data dengan benar.")
            else:
                user_data = scaler.transform([[usia, berat, tinggi]])
                prediction = model.predict(user_data)
                predicted_label = encoder.inverse_transform(prediction)[0]
                
                if predicted_label == "Gizi Baik":
                    color = "green"
                elif predicted_label == "Risiko Gizi Lebih":
                    color = "orange"
                else:
                    color = "red"
                
                st.markdown(f"<h3 style='color: {color};'>Hasil Prediksi: {predicted_label}</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
