#!/bin/bash
# Script untuk menginstal library Python yang dibutuhkan untuk Streamlit

# Menginstal pip jika belum terinstal
if ! command -v pip &> /dev/null
then
    echo "pip belum terinstal. Menginstal pip..."
    sudo apt-get install python3-pip
fi

# Menginstal library yang diperlukan
pip install streamlit pandas numpy matplotlib scikit-learn
