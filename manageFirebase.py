import firebase_admin
from firebase_admin import credentials, firestore, auth
import datetime
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Create a new user using email and password (Firebase Authentication)
def create_user_with_email(email: str, password: str):
    try:
        user = auth.create_user(email=email, password=password)
        print(f"Successfully created new user: {user.uid}")
        return user.uid
    except Exception as e:
        print(f"Error creating user: {e}")
        return None

# Initialize Firebase app using the service account key file
def initialize_firebase(service_account_path: str):
    try:
        cred = credentials.Certificate(service_account_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            print("Firebase has been successfully initialized.")
    except Exception as e:
        print(f"Error initializing Firebase: {e}")

# Get Firestore database client
def get_firestore_client():
    return firestore.client()

# Determine if a given emotion and transcript indicate depression
def is_depressive(emotion: str, transcript: str) -> bool:
    depressive_emotions = ["sad", "fear", "angry"]
    keywords = ["hopeless", "worthless", "tired", "give up", "can't take it", "nothing matters"]

    # Check if emotion is depressive and any keyword is found in the transcript
    if emotion.lower() in depressive_emotions:
        for word in keywords:
            if word in transcript.lower():
                return True
    return False

# Store an emotion log in Firestore and flag user if depressive symptoms detected
def store_emotion_log(user_id: str, emotion: str, response: str, transcript: str):
    db = get_firestore_client()
    timestamp = datetime.datetime.now()

    log_data = {
        "emotion": emotion,
        "response": response,
        "transcript": transcript,
        "timestamp": timestamp
    }

    # Save the log to Firestore under users/{user_id}/logs
    db.collection("users").document(user_id).collection("logs").add(log_data)
    print(f"Emotion log stored for {user_id}: {log_data}")

    # If depressive signs are detected, mark the user as flagged
    if is_depressive(emotion, transcript):
        db.collection("users").document(user_id).set({"flagged": True}, merge=True)
        print(f"User {user_id} flagged for depressive symptoms.")

# Check if the user is flagged for depressive symptoms
def is_user_flagged(user_id: str) -> bool:
    db = get_firestore_client()
    try:
        user_doc = db.collection("users").document(user_id).get()
        if user_doc.exists:
            data = user_doc.to_dict()
            return data.get("flagged", False)
        return False
    except Exception as e:
        print(f"Error checking flagged status: {e}")
        return False

# Retrieve all emotion logs for a user from Firestore
def get_emotion_logs(user_id: str):
    db = get_firestore_client()
    logs = []
    try:
        docs = db.collection("users").document(user_id).collection("logs").stream()
        for doc in docs:
            logs.append(doc.to_dict())
        return logs
    except Exception as e:
        print(f"Error retrieving logs for {user_id}: {e}")
        return []

# Streamlit UI helper to visualize emotion logs as a bar chart
def display_emotion_chart(user_id: str, st):
    logs = get_emotion_logs(user_id)
    if not logs:
        st.info("No logs available to display.")
        return

    # Count emotion occurrences

    emotions = [log["emotion"] for log in logs if "emotion" in log]
    emotion_counts = Counter(emotions)

    df = pd.DataFrame.from_dict(emotion_counts, orient="index", columns=["Count"])
    df = df.sort_values(by="Count", ascending=False)

    st.subheader("Emotion History Overview")
    st.bar_chart(df)

# Streamlit UI helper to show flagged message for users in distress
def display_flagged_warning(user_id: str, st):
    if is_user_flagged(user_id):
        st.warning("""
        It looks like you're experiencing emotional distress.
        Please consider reaching out for help.
        Dr. Emily Kim – Mental Health Specialist
        010-1234-5678 | support.example.com
        """)
        
# Code for testing 
if __name__ == "__main__":
    initialize_firebase("config/FirebaseDB.json")

    test_user = "test_user_001"
    test_emotion = "sad"
    test_response = "I'm sorry you're feeling that way. You're not alone."
    test_transcript = "I feel tired and worthless lately."

    store_emotion_log(
        user_id=test_user,
        emotion=test_emotion,
        response=test_response,
        transcript=test_transcript
    )

    flagged = is_user_flagged(test_user)
    print("Is user flagged?:", flagged)

    logs = get_emotion_logs(test_user)
    print("Retrieved logs:")
    for log in logs:
        print(log)
