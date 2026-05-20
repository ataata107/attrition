import streamlit as st
from agent import build_agent

st.set_page_config(
    page_title="HR Attrition Agent",
    page_icon="👥",
    layout="wide",
)

st.title("IBM HR Analytics — Attrition Q&A Agent")
st.caption("Ask any question about the employee dataset or attrition patterns.")

with st.sidebar:
    st.header("About")
    st.markdown("""
**Data sources:**
- Raw IBM HR dataset (1,470 employees)
- Attrition pattern mining results (87,114 segments)

**Example questions:**
- Which department has the highest attrition rate?
- What is the average monthly income of employees who left?
- Show the top 5 highest-risk employee segments
- How does overtime affect attrition for sales reps?
- What percentage of young employees with low income left?
""")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "agent" not in st.session_state:
    with st.spinner("Loading datasets and initialising agent..."):
        st.session_state.agent = build_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about employee attrition..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.invoke({"input": prompt})
                answer = response["output"]
            except Exception as e:
                answer = f"Sorry, I ran into an error: {str(e)}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
