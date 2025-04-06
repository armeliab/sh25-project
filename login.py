import streamlit as st
from firebase_admin import auth
from manageFirebase import create_user_with_email, initialize_firebase

# Initialize Firebase only once
if "firebase_initialized" not in st.session_state:
    initialize_firebase("config/FirebaseDB.json")
    st.session_state["firebase_initialized"] = True

def login_page():
    if "user_id" in st.session_state and "user_email" in st.session_state:
        st.success(f"Logged in as: {st.session_state['user_email']}")
        if st.button("Logout"):
            del st.session_state["user_id"]
            del st.session_state["user_email"]
            st.rerun()
        return st.session_state["user_id"]

    st.title("Login or Sign Up")

    mode = st.radio("Choose mode", ["Login", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if mode == "Sign Up":
        if st.button("Create Account"):
            user_id = create_user_with_email(email, password)
            if user_id:
                st.success("Account created. Please log in to continue.")

    elif mode == "Login":
        if st.button("Login"):
            try:
                user = auth.get_user_by_email(email)
                st.session_state["user_id"] = user.uid
                st.session_state["user_email"] = user.email  
                st.success(f"Logged in as: {email}")
                st.rerun()
            except auth.UserNotFoundError:
                st.error("Login failed. User not found.")
            except Exception as e:
                st.error(f"Login failed: {e}")

    return None
