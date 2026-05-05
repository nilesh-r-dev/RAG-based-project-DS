import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests


# def create_embedding(text_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })

#     embedding = r.json()["embeddings"] 
#     return embedding


def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "bge-m3",
        "prompt": text
    })

    return r.json()["embedding"]   

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    # print(response)
    return response

df = joblib.load('embeddings.joblib')
df = df[df['embedding'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

df['embedding'] = df['embedding'].apply(
    lambda x: x[0] if (isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)) else x
)
# df['embedding'] = df['embedding'].apply(lambda x: x[0] if isinstance(x[0], list) else x)
# df = df[df['embedding'].apply(lambda x: len(x) > 0)]


incoming_query = input("Ask a Question: ")
#question_embedding = create_embedding([incoming_query])[0] 
# question_embedding = create_embedding(incoming_query)
question_embedding = create_embedding(incoming_query)

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx] 
# print(new_df[["title", "number", "text"]])

# prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

# {new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
# ---------------------------------
# "{incoming_query}"
# User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
# '''

new_df = df.loc[max_indx]

# ✅ REPLACE BELOW THIS LINE (your old prompt)
prompt = f'''
You are a helpful assistant for a web development course.

You are given video subtitle chunks in JSON format:
Each chunk has:
- title
- video number
- start time
- end time
- text

DATA:
{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

---------------------------------

User Question: "{incoming_query}"

IMPORTANT INSTRUCTIONS:
1. You MUST mention:
   - Video number
   - Timestamp (start and end)
2. Answer ONLY from the given data
3. Format your answer like this:

Video: <number>
Time: <start> to <end>
Explanation: <your answer>

4. If multiple matches, show top 2–3 results
5. If not found, say: "This question is not covered in the course"

DO NOT give general answers.
DO NOT ignore timestamps.

Answer now:
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)
# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])
