import streamlit as st
import requests

API_URL = "https://hamid-nawaz-gct-ena-obe-assistant.hf.space"

st.set_page_config(
    page_title="ENA OBE Assistant",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ ENA OBE Assistant")
st.markdown("*RAG-powered chatbot and OBE question generator for Electrical Network Analysis*")

tab1, tab2 = st.tabs(["💬 Chat", "📝 Question Generator"])

# ── Chat Tab ──────────────────────────────────────────
with tab1:
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question:")
    if st.button("Ask"):
        if question:
            with st.spinner("Thinking..."):
                r = requests.post(f"{API_URL}/chat/", json={"question": question})
                if r.status_code == 200:
                    data = r.json()
                    st.markdown("### Answer")
                    st.write(data["answer"])
                    st.caption(f"Sources: {', '.join(data['sources'])}")
                else:
                    st.error(f"Error: {r.status_code}")

# ── Question Generator Tab ────────────────────────────
with tab2:
    st.subheader("Generate OBE Questions")
    col1, col2 = st.columns(2)
    with col1:
        clo    = st.text_input("CLO", "CLO1: Apply KVL to analyze DC circuits")
        bloom  = st.selectbox("Bloom's Level", ["Remember (C1)", "Understand (C2)", "Apply (C3)", "Analyze (C4)"])
        topic  = st.text_input("Topic", "KVL")
    with col2:
        q_type = st.selectbox("Question Type", ["MCQ", "Short", "Long"])
        num_q  = st.slider("Number of Questions", 1, 10, 2)
        marks  = st.number_input("Marks per Question", 1, 20, 5)

    if st.button("Generate Questions"):
        with st.spinner("Generating..."):
            payload = {
                "clo": clo, "bloom": bloom, "q_type": q_type,
                "num_questions": num_q, "marks": int(marks), "topic": topic
            }
            r = requests.post(f"{API_URL}/questions/", json=payload)
            if r.status_code == 200:
                questions = r.json()["questions"]
                for q in questions:
                    st.markdown(f"**Q{q['q_no']}: {q['question']}**")
                    if "options" in q:
                        for opt in q["options"]:
                            st.write(opt)
                        st.success(f"✅ Answer: {q['answer']}")
                    st.divider()
            else:
                st.error(f"Error: {r.status_code}")