import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

training_data = [
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("how are you", "greeting"),
    ("bye", "goodbye"),
    ("goodbye", "goodbye"),
    ("thanks", "thanks"),
    ("thank you", "thanks"),
    ("what is ai", "ai"),
    ("define ai", "ai"),
]

responses = {
    "greeting": ["Hello!", "Hi there!", "Hey!"],
    "goodbye": ["Bye!", "See you!", "Take care"],
    "thanks": ["You're welcome!", "No problem"],
    "ai": ["AI means machines that think and learn like humans"]
}

X = [data[0] for data in training_data]
y = [data[1] for data in training_data]

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

print("🤖 Chatbot ready! Type 'exit' to stop.")

while True:
    user_input = input("You: ").lower()
    if user_input == "exit":
        break

    user_vector = vectorizer.transform([user_input])
    intent = model.predict(user_vector)[0]

    print("Bot:", random.choice(responses[intent]))
