import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.ticker as ticker

# Tema aplikasi
st.set_page_config(page_title="Peramalan Hasil Panen Padi dengan Metode Forecasting Extreme Learning Machine (ELM)", layout="wide", initial_sidebar_state="expanded")

# Fungsi untuk memuat dataset
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv('dataset_bkl.csv')

# Fungsi untuk deteksi dan penghapusan outlier
def remove_outliers(data):
    X = data.drop(['tahun', 'produksi', 'produktivitas'], axis=1, errors='ignore')
    y = data['produksi']

    # Deteksi outlier menggunakan LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    outliers = lof.fit_predict(X)

    # Indeks outliers
    outliers_indices = np.where(outliers == -1)[0]

    # Menghapus outliers dari data
    data_no_outliers = data.drop(index=outliers_indices)

    return data_no_outliers, outliers_indices

# Fungsi untuk menampilkan visualisasi data
def plot_data(y, title='Produksi Padi', color='blue'):
    plt.figure(figsize=(10, 6))
    plt.plot(y, marker='o', linestyle='-', color=color, label='Data Asli')
    plt.title(title, weight='bold')
    plt.xlabel('Index', weight='bold')
    plt.ylabel('Produksi (kw)', weight='bold')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Judul aplikasi
st.title('🌾 Peramalan Hasil Panen Padi dengan Metode Forecasting Extreme Learning Machine (ELM) 🌾')

# Upload file
uploaded_file = st.file_uploader("📤 Upload file CSV (opsional)", type=["csv"])

# Membuat tab
tab1, tab2, tab3 = st.tabs(["📊 Dataset", "🔍 Deteksi Outlier", "🔮 Prediksi"])

# Tab untuk Dataset
with tab1:
    data = load_data(uploaded_file)
    
    if 'tahun' in data.columns:
        data['tahun'] = data['tahun'].astype(str)

    st.subheader("📋 Data Awal")
    st.dataframe(data, use_container_width=True)

# Tab untuk Deteksi Outlier
with tab2:
    st.subheader("🔍 Deteksi Outlier Menggunakan LOF")

    # Deteksi outlier
    data_no_outliers, outliers_indices = remove_outliers(data)

    st.write(f"Jumlah data awal: {len(data)}")
    st.write(f"Jumlah data setelah menghapus outliers: {len(data_no_outliers)}")

    # Visualisasi data dengan outliers
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(data)), data['produksi'], label='Data Asli', color='blue')
    if len(outliers_indices) > 0:
        plt.scatter(outliers_indices, data.iloc[outliers_indices]['produksi'], color='red', label='Outliers')
    plt.title('Deteksi Outlier dengan LOF', weight='bold')
    plt.xlabel('Indeks Data', weight='bold')
    plt.ylabel('Produksi (kw)', weight='bold')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

    st.subheader("📋 Data Setelah Menghapus Outlier")
    st.dataframe(data_no_outliers, use_container_width=True)

# Tab untuk Normalisasi
with tab3:
    # Ambil data produksi
    y = data_no_outliers['produksi'].values
    
    # Normalisasi MinMax
    scaler = MinMaxScaler()
    y_normalized = scaler.fit_transform(y.reshape(-1, 1))

    # Visualisasi data sebelum normalisasi
    st.subheader("📉 Visualisasi Data Sebelum Normalisasi")
    plot_data(y, title='Produksi Padi Sebelum Normalisasi', color='blue')
    
    # Visualisasi data setelah normalisasi
    st.subheader("📈 Visualisasi Data Setelah Normalisasi")
    plot_data(y_normalized, title='Produksi Padi Setelah Normalisasi', color='green')

# Tab untuk Prediksi
with tab3:
    # Membagi data menjadi data latih dan data uji
    X = np.arange(len(y)).reshape(-1, 1)
    train_size = int(len(X) * 0.9)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_normalized[:train_size], y_normalized[train_size:]

    # Inisialisasi bobot dari file .npy
    W = np.load('weights.npy')

    # Hitung hidden layer
    Hinit = X_train @ W.T
    H = 1 / (1 + np.exp(-Hinit))

    # Hitung beta menggunakan pseudoinverse
    beta = np.linalg.pinv(H) @ y_train

    # Prediksi di data latih dan data uji
    y_train_pred = H @ beta
    Hinit_test = X_test @ W.T
    H_test = 1 / (1 + np.exp(-Hinit_test))
    y_test_pred = H_test @ beta

    # Prediksi 3 tahun ke depan
    years_ahead = 3 
    X_future = np.array([[len(y) + i] for i in range(1, years_ahead + 1)])
    Hinit_future = X_future @ W.T
    H_future = 1 / (1 + np.exp(-Hinit_future))
    output_future = H_future @ beta
    output_future_rounded = scaler.inverse_transform(output_future.reshape(-1, 1)).flatten()

    # Hasil prediksi
    st.write("🔮 Prediksi Produksi", years_ahead, "Tahun Ke Depan:", output_future_rounded)

    # Hitung MAPE
    def calculate_mape(y_actual, y_pred):
        return np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

    # Denormalisasi nilai y_test dan y_test_pred
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_actual = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    # Membuat tabel gabungan untuk data training, testing, dan prediksi 3 tahun ke depan (untuk visualisasi)
    train_data = pd.DataFrame({
        'Tipe Data': ['Training'] * len(y_train),
        'Produksi Aktual': scaler.inverse_transform(y_train.reshape(-1, 1)).flatten(),
        'Produksi Prediksi': scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
    })

    test_data = pd.DataFrame({
        'Tipe Data': ['Testing'] * len(y_test),
        'Produksi Aktual': y_test_actual,
        'Produksi Prediksi': y_test_pred_actual
    })

    future_data = pd.DataFrame({
        'Tipe Data': ['Prediksi'] * len(X_future),
        'Produksi Aktual': [np.nan] * len(X_future),
        'Produksi Prediksi': output_future_rounded
    })

    combined_data = pd.concat([train_data, test_data, future_data], ignore_index=True)

    st.subheader("📈 Tabel Hasil Prediksi")
    st.dataframe(combined_data, use_container_width=True)


    # Grafik
    plt.figure(figsize=(25, 15), facecolor='#e6e6e6') 

    # Mengatur tahun untuk data pelatihan dan pengujian
    tahun_awal = int(data_no_outliers['tahun'].values[0])
    tahun_train = np.arange(tahun_awal, tahun_awal + len(train_data))
    tahun_terakhir_train = tahun_train[-1]
    tahun_test = np.arange(tahun_terakhir_train + 1, tahun_terakhir_train + 1 + len(test_data))
    tahun_future = np.arange(tahun_terakhir_train + len(test_data) + 1, tahun_terakhir_train + len(test_data) + 1 + len(future_data))

    # Plot data
    plt.plot(tahun_train, train_data['Produksi Aktual'], marker='o', linestyle='--', label='Produksi Aktual (Training)', color='blue')
    plt.plot(tahun_test, test_data['Produksi Aktual'], marker='o', linestyle='--', label='Produksi Aktual (Testing)', color='orange')
    plt.plot(tahun_test, test_data['Produksi Prediksi'], marker='o', linestyle='-', label='Produksi Prediksi (Testing)', color='green')
    plt.plot(tahun_future, future_data['Produksi Prediksi'], marker='o', linestyle='-', label='Produksi Prediksi (Masa Depan)', color='red')

    plt.title('Visualisasi Produksi Padi: Aktual dan Prediksi', weight='bold', color='#333')
    plt.xlabel('Tahun', weight='bold', color='#333')
    plt.ylabel('Produksi (kw)', weight='bold', color='#333')
    plt.grid(True)
    plt.legend()

    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(250000))

    # Data label
    for i, v in enumerate(train_data['Produksi Aktual']):
        plt.text(tahun_train[i], v + 5, str(int(v)), color='blue', ha='center', fontsize=10) 
    for i, v in enumerate(test_data['Produksi Aktual']):
        plt.text(tahun_test[i], v + 5, str(int(v)), color='orange', ha='center', fontsize=10)
    for i, v in enumerate(test_data['Produksi Prediksi']):
        plt.text(tahun_test[i], v + 5, str(int(v)), color='green', ha='center', fontsize=10)
    for i, v in enumerate(future_data['Produksi Prediksi']):
        plt.text(tahun_future[i], v + 5, str(int(v)), color='red', ha='center', fontsize=10)

    # Display the plot
    st.pyplot(plt)
