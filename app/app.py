from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
# Load the pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)

def get_corpus():
    corpus = []
    parquet_file = "../data/0000.parquet"
    df = pd.read_parquet(parquet_file, engine='pyarrow')
    for idx,rows in df.iterrows():
        if idx > 20:
            break
        for sent in rows['set']:
           corpus.append(sent)
    return corpus

# Define a function to calculate cosine similarity
def get_cosine_similarity(query, corpus):
  query_embedding = model.encode(query)
  corpus_embeddings = model.encode(corpus)
  return util.cos_sim(query_embedding, corpus_embeddings).tolist()[0]

@app.route("/", methods=["GET", "POST"])
def index():
  if request.method == "POST":
    query = request.form["query"]
    corpus = get_corpus()
    # corpus = ["This is a sample sentence.", "This sentence is similar to the first one.", "This one is a bit different."]
    similarity = get_cosine_similarity(query, corpus)
    index_max = np.argmin(similarity)
    max_sim = np.max(similarity)
    return render_template("index.html", query=query, similarity={"similarity":max_sim, "most_similar_sent":corpus[index_max]})
    
  else:
    return render_template("index.html")

if __name__ == "__main__":
  app.run(debug=True)
