import firebase_admin
from firebase_admin import credentials, firestore, auth
import datetime
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

firebase_initialized = False

def initialize_firebase(service_account_path: str):
    global firebase_initialized
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
            firebase_initialized = True
            print("Firebase initialized.")
        except Exception as e:
            print(f"Error initializing Firebase: {e}")

def create_user_with_email(email: str, password: str):
    if not firebase_admin._apps:
        initialize_firebase("../config/FirebaseDB.json")
    try:
        user = auth.create_user(email=email, password=password)
        st.success(f"User created: {user.uid}")
        # Store user basic profile
        db = get_firestore_client()
        db.collection("users").document(user.uid).set({
            "email": email,
            "created_at": datetime.datetime.now()
        }, merge=True)
        return user.uid
    except Exception as e:
        st.error(f"Failed to create user: {str(e)}")
        return None

def get_firestore_client():
    return firestore.client()

def is_depressive(emotion: str, transcript: str) -> bool:
    emotion = (emotion or "").lower()
    transcript = (transcript or "").lower()

    depressive_emotions = [
        "sad", "sadness", "depressed", "depression", "pain", "disappointment",
        "fear", "fearful", "anxious", "anxiety", "nervous",
        "worthless", "hopeless", "frustrated", "empty", "lonely", "tired"
    ]

    depressive_keywords = [
        "give up", "can't take it", "nothing matters",
        "hate myself", "why bother", "no point", "suicidal", "i'm done"
    ]

    if emotion in depressive_emotions:
        return True

    return any(kw in transcript for kw in depressive_keywords)

def store_emotion_log(user_id: str, emotion: str, response: str, transcript: str):
    db = get_firestore_client()
    timestamp = datetime.datetime.now()

    log_data = {
        "emotion": emotion,
        "response": response,
        "transcript": transcript,
        "timestamp": timestamp
    }

    db.collection("users").document(user_id).collection("logs").add(log_data)
    print(f"Emotion log stored for {user_id}: {log_data}")

    if is_depressive(emotion, transcript):
        db.collection("users").document(user_id).set({"flagged": True}, merge=True)
        print(f"User {user_id} flagged for depressive symptoms.")

def is_user_flagged(user_id: str) -> bool:
    db = get_firestore_client()
    try:
        user_doc = db.collection("users").document(user_id).get()
        return user_doc.to_dict().get("flagged", False) if user_doc.exists else False
    except Exception as e:
        print(f"Error checking flagged status: {e}")
        return False

def get_emotion_logs(user_id: str):
    db = get_firestore_client()
    try:
        docs = db.collection("users").document(user_id).collection("logs").stream()
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        print(f"Error retrieving logs for {user_id}: {e}")
        return []

def display_emotion_chart(user_id: str, st):
    logs = get_emotion_logs(user_id)
    if not logs:
        st.info("No logs available to display.")
        return

    emotions = [log["emotion"] for log in logs if "emotion" in log]
    df = pd.DataFrame.from_dict(Counter(emotions), orient="index", columns=["Count"]).sort_values("Count", ascending=False)

    st.subheader("Emotion History Overview")
    st.bar_chart(df)

def display_flagged_warning(user_id: str, st):
    if is_user_flagged(user_id):
        st.warning("""
**Emotional distress detected.**

You may be experiencing signs of emotional distress. Please consider talking to a professional.

Dr. Emily Kim — Mental Health Specialist  
Phone: 010-1234-5678  
Website: [support.example.com](https://support.example.com)
""")
