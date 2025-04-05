import firebase_admin
from firebase_admin import credentials, firestore
import datetime

# Function to initialize Firebase using a service account JSON file
def initialize_firebase(service_account_path: str):
    """
    Initialize Firebase with the provided service account JSON file.
    
    Args:
        service_account_path (str): Path to the Firebase service account JSON file.
    """
    try:
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
        print("Firebase has been successfully initialized.")
    except Exception as e:
        print(f"Error initializing Firebase: {e}")

# Function to get a Firestore client instance after Firebase is initialized
def get_firestore_client():
    """
    Retrieve the Firestore client.
    
    Returns:
        firestore.Client: An instance of Firestore client.
    """
    return firestore.client()

# Function to store an emotion log into Firestore
def store_emotion_log(user_id: str, emotion: str, response: str):
    """
    Store an emotion log in Firestore.
    
    Args:
        user_id (str): Identifier for the user.
        emotion (str): Detected emotion (e.g., "happy", "sad", "angry").
        response (str): The response text generated based on the emotion.
    """
    db = get_firestore_client()
    timestamp = datetime.datetime.now()
    # Create a document ID using the user ID and current timestamp
    doc_id = f"{user_id}_{timestamp.strftime('%Y%m%d%H%M%S')}"
    data = {
        "user_id": user_id,
        "emotion": emotion,
        "response": response,
        "timestamp": timestamp
    }
    db.collection("emotion_logs").document(doc_id).set(data)
    print(f"Emotion log stored: {data}")

# Function to retrieve emotion logs from Firestore
def get_emotion_logs():
    """
    Retrieve all emotion logs from the Firestore 'emotion_logs' collection.
    
    Returns:
        list: A list of dictionaries, each representing an emotion log.
    """
    db = get_firestore_client()
    logs = []
    try:
        docs = db.collection("emotion_logs").stream()
        for doc in docs:
            logs.append(doc.to_dict())
        return logs
    except Exception as e:
        print(f"Error retrieving logs: {e}")
        return []

# Module testing section
if __name__ == "__main__":
    # Replace with the actual path to your Firebase service account JSON file
    service_account_path = "config/FirebaseDB.json"
    
    # Initialize Firebase
    initialize_firebase(service_account_path)
    
    # Example: Store an emotion log (using a sample user, emotion, and response)
    store_emotion_log("demo_user", "happy", "You sound happy! Keep smiling!")
    
    # Example: Retrieve and print stored emotion logs
    logs = get_emotion_logs()
    for log in logs:
        print(log)
