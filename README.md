from tkinter import messagebox, filedialog, Tk
import pandas as pd
import re
import unicodedata
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math

# Função para limpar e normalizar texto
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[\[\]']", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    text = ''.join(
        char for char in unicodedata.normalize('NFD', text)
        if unicodedata.category(char) != 'Mn'
    )
    return text

# Função de recomendação com base em sintomas do usuário
def recommend_medicines(user_input, medicine_df, bow_matrix, symptom_index, unique_symptoms):
    user_input = preprocess_text(user_input)
    user_vector = np.zeros(len(unique_symptoms))

    for word in user_input.split():
        if word in symptom_index:
            user_vector[symptom_index[word]] = 1

    similarities = cosine_similarity([user_vector], bow_matrix.values)[0]
    angles = [
        round(math.degrees(math.acos(sim)), 1) if -1 <= sim <= 1 else None
        for sim in similarities
    ]

    medicine_df = medicine_df.copy()
    medicine_df['similarity'] = similarities
    medicine_df['angle_degrees'] = angles

    recommended = medicine_df.sort_values(by='similarity', ascending=False).head(10)
    return recommended[['Name', 'Indication', 'Category', 'Dosage Form', 'similarity', 'angle_degrees']]

# Carregar o dataset de medicamentos
"""root = Tk()
root.withdraw()
caminho_arquivo = filedialog.askopenfilename(title="Selecione o arquivo CSV", filetypes=[("CSV", "*.csv")])"""
arquivo = r'C:\Users\irisd\iCloudDrive\IRIS\Python\Nova pasta (3)\medicine_dataset.csv'
if arquivo:
    medicine_df = pd.read_csv(arquivo)
    medicine_df.head()
else:
    print("Nenhum arquivo selecionado!")

# Preprocessar a coluna "Indication"
medicine_df['cleaned_indication'] = medicine_df['Indication'].apply(preprocess_text)

# Construir o vocabulário (Bag of Words)
unique_symptoms = sorted(set(" ".join(medicine_df['cleaned_indication']).split()))
symptom_index = {symptom: idx for idx, symptom in enumerate(unique_symptoms)}

bow_matrix = pd.DataFrame(0, index=medicine_df.index, columns=unique_symptoms)
for idx, indication in enumerate(medicine_df['cleaned_indication']):
    for word in indication.split():
        if word in bow_matrix.columns:
            bow_matrix.loc[idx, word] = 1

# Entrada do usuário
user_input = input("Descreva o(s) sintoma(s): ")  # Ex: "dor infecção ferida"
recommendations = recommend_medicines(user_input, medicine_df, bow_matrix, symptom_index, unique_symptoms)

# Exibir as recomendações
print("\nMedicamentos recomendados:")
print(recommendations.to_string(index=False))
