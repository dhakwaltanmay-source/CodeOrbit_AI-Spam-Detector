# Sample dataset
messages = [
    "Congratulations! You won a free prize. Claim now.",
    "Win a free iPhone by clicking this link.",
    "You have won a lottery worth 10 lakh rupees.",
    "Get a free gift card. Click here now.",
    "Urgent! Claim your cash reward immediately.",
    "You are selected for a free vacation.",

    "Hey, are we meeting today?",
    "Please call me when you are free.",
    "Your assignment is due tomorrow.",
    "Let's have lunch together.",
    "I will reach home in 10 minutes.",
    "Thank you for your help."
]

# Labels: 1 = Spam, 0 = Not Spam
labels = [
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0
]

print("Dataset loaded successfully!")
print("Total messages:", len(messages))
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text messages into numerical features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(messages)

print("\nText converted into numerical features!")
print("Feature matrix shape:", X.shape)
from sklearn.naive_bayes import MultinomialNB

# Create the AI model
model = MultinomialNB()

# Train the model using our dataset
model.fit(X, labels)

print("\nSpam detection model trained successfully!")
# Function to predict message
def predict_message(message):

    # Convert the input message into numerical features
    message_vector = vectorizer.transform([message])

    # Predict using the trained model
    prediction = model.predict(message_vector)

    if prediction[0] == 1:
        return "SPAM"
    else:
        return "NOT SPAM"


# Test example messages
print("\nMessage Predictions:")

test_messages = [
    "Congratulations! You won a free prize!",
    "Hey, are you coming to college today?",
    "Claim your cash reward now!"
]

for msg in test_messages:
    result = predict_message(msg)
    print(f"Message: {msg}")
    print(f"Prediction: {result}")
    print("-" * 40)
    # Interactive spam detection application

print("\n===================================")
print("       AI SPAM MESSAGE DETECTOR")
print("===================================")

while True:
    user_message = input("\nEnter a message (or type 'exit' to quit): ")

    if user_message.lower() == "exit":
        print("Thank you for using Spam Detector!")
        break

    if not user_message.strip():
        print("Please enter a valid message.")
        continue

    result = predict_message(user_message)

    print("\nPrediction:", result)
    print("-" * 40)