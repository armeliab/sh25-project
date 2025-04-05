import streamlit as st
from manageFirebase import create_user_with_email

def login_page():
    st.title("Login / Sign Up")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login or Sign Up"):
        user_id = create_user_with_email(email, password)
        if user_id:
            st.success(f"Logged in as: {email}")
            st.session_state["user_id"] = user_id
        else:
            st.warning("Login failed or user already exists.")

    return st.session_state.get("user_id", None)
