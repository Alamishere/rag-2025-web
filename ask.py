import json
import numpy as np
import faiss
import boto3

# Load FAISS index and texts
index = faiss.read_index("faiss_index.index")
with open("texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

# Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def get_titan_embedding(text):
    body = json.dumps({
        "inputText": text
    })
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        body=body
    )
    result = json.loads(response["body"].read())
    return np.array(result['embedding'], dtype='float32')

def ask_question(query):
    # Step 1: Embed query and search index
    query_vec = get_titan_embedding(query)
    D, I = index.search(np.array([query_vec]), k=3)
    context = "\n\n".join(texts[i] for i in I[0])

    # Step 2: Build Claude 3.5 compatible message payload
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 400,
        "temperature": 0.3,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are a helpful assistant.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question:\n{query}"
                )
            }
        ]
    }

    # Step 3: Call Claude 3.5 Sonnet
    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(payload),
            contentType="application/json"
        )
        result = json.loads(response["body"].read())
        print("\n🧠 Answer:\n")
        print(result["content"][0]["text"].strip())
    except Exception as e:
        print("⚠️ Bedrock call failed:")
        print(e)

# Run from terminal
if __name__ == "__main__":
    query = input("Ask a question: ")
    ask_question(query)
