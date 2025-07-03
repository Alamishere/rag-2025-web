import json
import numpy as np
import faiss
import boto3
import streamlit as st

# Load FAISS index and texts
@st.cache_resource
def load_resources():
    index = faiss.read_index("faiss_index.index")
    with open("texts.json", "r", encoding="utf-8") as f:
        texts = json.load(f)
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    return index, texts, bedrock

index, texts, bedrock = load_resources()

def get_titan_embedding(text):
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        body=body
    )
    result = json.loads(response["body"].read())
    return np.array(result['embedding'], dtype='float32')

def ask_question(query, conversation_history):
    if not query.strip():
        return "Please enter a question."
    
    # Get context from FAISS
    query_vec = get_titan_embedding(query)
    D, I = index.search(np.array([query_vec]), k=3)
    context = "\n\n".join(texts[i] for i in I[0])

    # Build conversation with history
    messages = []
    
    # Add conversation history (already alternating)
    for msg in conversation_history:
        if msg.get("content", "").strip():
            messages.append(msg)
    
    # Add current query with context
    current_query = f"Context:\n{context}\n\nQuestion: {query}\n\nGive a direct answer. Do not start with phrases like 'According to the context', 'Based on the context', 'The context shows', or similar references."
    messages.append({"role": "user", "content": current_query})

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 400,
        "temperature": 0.3,
        "messages": messages
    }

    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(payload),
            contentType="application/json"
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("ALAMS RAG DEMO")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input form
with st.form("chat_form"):
    prompt = st.text_input("")
    submitted = st.form_submit_button("Send")

# Process input
if submitted and prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get assistant response
    with st.spinner("Thinking..."):
        response = ask_question(prompt, st.session_state.messages[:-1])
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history
for message in reversed (st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Clear chat button at bottom
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()