from flask import Flask,request,render_template,redirect,url_for
app=Flask(__name__)

import pickle

# Load the saved TF-IDF pipeline
with open('tfidf_pipeline.pkl', 'rb') as tfidf_file:
    tfidf_pipeline = pickle.load(tfidf_file)

# Load the saved Logistic Regression model
with open('logreg_model.pkl', 'rb') as model_file:
    logreg_model = pickle.load(model_file)
@app.route('/')
def home():
    return render_template('index.html')  # Render the user input form

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form.get('review')
        rating = request.form.get('rating')

        if not review or not rating:
            return render_template('index.html', error="Please provide both review and rating.")

        # Preprocess and vectorize the review
        review_vectorized = tfidf_pipeline['tfidf_vectorizer'].transform([review])

        # Predict sentiment
        sentiment = logreg_model.predict(review_vectorized)[0]

        # Map sentiment to label
        sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment_label = sentiment_mapping[sentiment]

        # Return result to the user
        return render_template('index.html', 
                               review=review, 
                               rating=rating, 
                               predicted_sentiment=sentiment_label)

    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=True)



    
