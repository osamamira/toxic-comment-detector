from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

#uploading  model and vectorizer
model = joblib.load('toxic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# cleaning function
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    user_input = ""
    if request.method == "POST":
        user_input = request.form["comment"]
        cleaned = clean_text(user_input)
        X = vectorizer.transform([cleaned])
        prediction = model.predict(X)[0]
        result = "🚨 Toxic Comment" if prediction == 1 else "✅ Non-Toxic Comment"
    return render_template(r"index.html", result=result, comment=user_input)

if __name__ == "__main__":
    app.run(debug=True)