# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

app = Flask(__name__)

# Load and process the dataset
df = pd.read_csv("Spotify_final_dataset.csv", low_memory=False)[:1000]
df = df.drop_duplicates(subset="Song Name")
df = df.dropna(axis=0)
df = df.drop(df.columns[3:], axis=1)
df["data"] = df.apply(lambda value: " ".join(value.astype("str")), axis=1)

# Vectorize the data
vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df["data"])

# Fit the NearestNeighbors model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(vectorized)

def recommend_items(input_song, knn_model, vectorizer, df, top_n=5):
    input_vectorized = vectorizer.transform([input_song]).toarray()
    distances, indices = knn_model.kneighbors(input_vectorized, n_neighbors=top_n+1)
    recommended_songs = df.iloc[indices[0][1:]]["Song Name"]
    return recommended_songs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    input_song = request.form['song_name']
    if input_song in df["Song Name"].values:
        recommendation = recommend_items(input_song, knn_model, vectorizer, df)
        return jsonify({'recommendations': list(recommendation)})
    else:
        return jsonify({'error': 'Song not found in the database'})

if __name__ == '__main__':
    app.run(debug=True)
