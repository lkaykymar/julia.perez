import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import math

# Baixar os stopwords se necessário
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Carregar o dataset
df = pd.read_excel('Medicine_Details.xlsx')

# Remover duplicatas
df = df.drop_duplicates(subset='Medicine Name', keep='first').reset_index(drop=True)

# Função de recomendação
def recommend_medicines(user_input, top_n=10):
    stop_words = stopwords.words('portuguese') + stopwords.words('english')
    tfidf = TfidfVectorizer(stop_words=stop_words, max_df=0.7, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['Uses'].fillna(''))

    user_vector = tfidf.transform([user_input])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    similar_indices = similarities.argsort()[::-1][:top_n]

    recommendations = []
    for i in similar_indices:
        similarity = similarities[i]
        if similarity > 0:
            angle = math.degrees(math.acos(min(1.0, similarity)))  # Garante que o valor não ultrapasse 1
            recommendation = {
                'Medicine': df.iloc[i]['Medicine Name'],
                'Uses': df.iloc[i]['Uses'],
                'Similarity': round(similarity, 4),
                'Angle (°)': round(angle, 2)
            }
            recommendations.append(recommendation)

    return recommendations

# Execução principal
if __name__ == "__main__":
    symptom = input("Enter the symptom you are experiencing: ")
    results = recommend_medicines(symptom)

    if results:
        print("\nRecommended medicines based on your symptom:\n")
        for rec in results:
            print(f"Medicine: {rec['Medicine']}\nUses: {rec['Uses']}\nSimilarity: {rec['Similarity']}\nAngle: {rec['Angle (°)']}°\n")
    else:
        print("No similar medicines found.")
