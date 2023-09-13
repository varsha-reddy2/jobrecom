from flask import Flask, render_template, request
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Create the Flask application
app = Flask(__name__)
model = joblib.load('jobrecom.pkl')

# Load your job data and create a TF-IDF vectorizer and matrix
job_data = pd.read_csv('IT_salaries.csv')  # Replace with your job data file
tfidf_vectorizer = TfidfVectorizer()
job_data = job_data.dropna(subset=['key_skills'])
tfidf_matrix = tfidf_vectorizer.fit_transform(job_data['key_skills'])

# Define the recommend function
def recommend(key_skills):
    # Transform user input skills to TF-IDF vector
    key_skills_tfidf = tfidf_vectorizer.transform([key_skills])

    # Calculate cosine similarity between user skills and job skills
    cosine_scores = cosine_similarity(key_skills_tfidf, tfidf_matrix)

    # Get job indices sorted by similarity
    job_indices = cosine_scores.argsort()[0][::-1]

    # Get top 5 job recommendations
    top_jobs = job_data['job_title'].iloc[job_indices][:5]

    return top_jobs.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    if request.method == 'POST':
        # Get skills input from the form
        key_skills = request.form.get('skills')

        # Get job recommendations using the 'recommend' function
        recommended_jobs = recommend(key_skills)

        # Return a valid response, for example, as a string
        return render_template('index.html', recommended_jobs=recommended_jobs)

if __name__ == '__main__':
    app.run(debug=True)
