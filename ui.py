import streamlit as st
import httpx
import os
import time

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8002/api/v1")

st.set_page_config(page_title="RAG PDF - AI Assistant", layout="wide")

st.title("📄 RAG PDF - AI Assistant")
st.markdown("Retrieval-Augmented Generation for Financial Documents")

# Authentication state
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None
if "sync_job_id" not in st.session_state:
    st.session_state["sync_job_id"] = None

# Sidebar for actions
with st.sidebar:
    st.header("Authentication")
    if not st.session_state["access_token"]:
        email = st.text_input("Email", value="admin@example.com")
        password = st.text_input("Password", type="password", value="strongadminpassword")
        if st.button("Login"):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/login/access-token",
                    data={"username": email, "password": password},
                    timeout=10
                )
                if response.status_code == 200:
                    st.session_state["access_token"] = response.json()["access_token"]
                    st.session_state["email"] = email
                    st.success("Logged in!")
                    st.rerun()
                else:
                    st.error("Login failed. Check credentials.")
            except Exception as e:
                st.error(f"Error during login: {e}")
    else:
        st.info(f"Logged in as {st.session_state.get('email', 'Admin')}")
        if st.button("Logout"):
            st.session_state["access_token"] = None
            st.rerun()

    if st.session_state["access_token"]:
        headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}
        
        st.divider()
        st.header("Actions")
        if st.button("🔄 Sync RBI Documents"):
            try:
                response = httpx.post(f"{API_BASE_URL}/ingest/rbi-sync", headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()["data"]
                    st.session_state["sync_job_id"] = data["job_id"]
                    st.success("Sync started in background!")
                    st.rerun()
                else:
                    st.error(f"Sync failed to start: {response.text}")
            except Exception as e:
                st.error(f"Error starting sync: {e}")

        if st.button("🔄 Sync NPCI Documents"):
            try:
                response = httpx.post(f"{API_BASE_URL}/ingest/npci-sync", headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()["data"]
                    st.session_state["sync_job_id"] = data["job_id"]
                    st.success("NPCI Sync started in background!")
                    st.rerun()
                else:
                    st.error(f"Sync failed to start: {response.text}")
            except Exception as e:
                st.error(f"Error starting NPCI sync: {e}")

        if st.session_state["sync_job_id"]:
            st.divider()
            st.header("Sync Progress")
            status_placeholder = st.empty()
            
            try:
                response = httpx.get(
                    f"{API_BASE_URL}/ingest/sync-status/{st.session_state['sync_job_id']}", 
                    headers=headers,
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()["data"]
                    status = data["status"]
                    synced = int(data["synced_files"])
                    total = int(data["total_files"])
                    errors = int(data["error_files"])
                    
                    if status == "running":
                        status_placeholder.info(f"Syncing: {synced}/{total} files (Errors: {errors})")
                        if total > 0:
                            st.progress(synced / total)
                        time.sleep(2)
                        st.rerun()
                    elif status == "completed":
                        status_placeholder.success(f"Sync completed! Total: {total}, Synced: {synced}, Errors: {errors}")
                        if st.button("Clear Status"):
                            st.session_state["sync_job_id"] = None
                            st.rerun()
                    else:
                        status_placeholder.error(f"Sync job failed: {data.get('message')}")
                        if st.button("Clear Status"):
                            st.session_state["sync_job_id"] = None
                            st.rerun()
                else:
                    st.error("Could not fetch sync status.")
            except Exception as e:
                st.error(f"Error fetching status: {e}")

        st.divider()
        st.header("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=["pdf", "docx", "html", "htm", "txt", "png", "jpg", "jpeg", "tiff", "bmp", "webp"]
        )
        if st.button("📤 Upload") and uploaded_file:
            with st.spinner("Uploading..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = httpx.post(f"{API_BASE_URL}/ingest/upload", headers=headers, files=files, timeout=300)
                    if response.status_code == 200:
                        st.success("Uploaded successfully!")
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Error during upload: {e}")

# Main Chat Interface
if st.session_state["access_token"]:
    st.subheader("💬 Ask a question")
    question = st.text_input("Enter your question about the indexed documents:")

    if st.button("🔍 Search & Generate"):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing documents..."):
                try:
                    headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}
                    response = httpx.post(
                        f"{API_BASE_URL}/query/", 
                        headers=headers,
                        json={"question": question}, 
                        timeout=120
                    )
                    if response.status_code == 200:
                        data = response.json()["data"]
                        st.markdown("### Answer")
                        st.write(data["answer"])
                        
                        with st.expander("Sources"):
                            if data.get("sources"):
                                for source in data["sources"]:
                                    st.write(f"- {source['file_name']} (Page {source['page']})")
                            else:
                                st.write("No specific sources found or mentioned in the answer.")
                    else:
                        st.error(f"Query failed: {response.text}")
                except Exception as e:
                    st.error(f"Error during query: {e}")
else:
    st.info("Please login from the sidebar to use the RAG features.")
    st.markdown("""
    ### Features:
    - **Sync RBI Documents**: Fetch latest circulars from RBI Scrapper.
    - **Custom Upload**: Upload your own PDF documents for indexing.
    - **QA Assistant**: Ask questions and get answers based on indexed content.
    """)
