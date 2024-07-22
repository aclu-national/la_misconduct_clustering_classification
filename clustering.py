''' 
Code to cluster text using TF-IDF, all-MiniLM-L6-v2, and DistilBERT base model (uncased) 
embeddings, Uniform Manifold Approximation and Projection (UMAP) dimensionality reduction,
and Hierarchical Density-Based Spatial Clustering of Applications with Noise (HBSCAN) clustering

For more information look below:
TF-IDF: https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/
all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
DistilBERT base model (uncased): https://huggingface.co/distilbert/distilbert-base-uncased
UMAP: https://arxiv.org/abs/1802.03426
HBSCAN: https://pberba.github.io/stats/2020/01/17/hdbscan/
'''

# -----------------------------------------------------------------------
# Author: Elijah Appelson
# Update Date: July 22nd, 2024
# -----------------------------------------------------------------------

# Importing Packages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import umap
import umap.plot
from sklearn.cluster import KMeans
import hdbscan
import plotly.express as px
import torch
from transformers import AutoTokenizer,AutoModel
import numpy as np
from sentence_transformers import SentenceTransformer

# Importing Data
allegation_link = "data_allegation.csv"
df = pd.read_csv(allegation_link)

# Defining stop words, lemmatization, and preprocess from 
# https://medium.com/@20pd11/1f8776a84ddb
def removeStopwords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lWords = [lemmatizer.lemmatize(word) for word in words]
    lText = ' '.join(lWords)
    return lText

def preprocess(text):
    text = str(text.split(':')[-1])
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\d_]+', '', text)
    text = removeStopwords(text)
    text = lemmatization(text)
    return text

# Creating our list of text data
textData = df['allegation'].dropna().unique().tolist()

# Cleaning our text data
textData_clean = [preprocess(text) for text in textData]

# ---------------------------- Tokenizing --------------------------------
# TF-IDF Tokenizing
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
text_tfidf = tfidf_vectorizer.fit_transform(textData_clean).toarray()

# all-MiniLM-L6-v2
model_minilm = SentenceTransformer('all-MiniLM-L6-v2')
text_minilm = model_minilm.encode(textData_clean)

# BERT
tokenizer_bert = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model_bert = AutoModel.from_pretrained("distilbert-base-uncased")

def encode(text_list):
    embeddings = []
    for text in text_list:
        inputs = tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model_bert(**inputs)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(sentence_embedding)
    return np.array(embeddings)

text_bert = encode(textData_clean)

# ------------------------ Embedding with UMAP --------------------------------

# Defining UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine')

# Fitting UMAP
embedding_tfidf = reducer.fit_transform(text_tfidf)
embedding_minilm = reducer.fit_transform(text_minilm)
embedding_bert = reducer.fit_transform(text_bert)

# ----------------------- Fitting hdbscan cluster -------------------------
# Defining hdbscan
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')

# Fitting hdbscan
hdbscan_labels_tfidf = hdbscan_clusterer.fit_predict(embedding_tfidf)
hdbscan_labels_minilm = hdbscan_clusterer.fit_predict(embedding_minilm)
hdbscan_labels_bert = hdbscan_clusterer.fit_predict(embedding_bert)

# Defining dataframes of clusters
cluster_tfidf = pd.DataFrame({
    'sentence': textData,
    'cluster': hdbscan_labels_tfidf,
    'x': embedding_tfidf[:, 0],
    'y': embedding_tfidf[:, 1]
})

cluster_minilm = pd.DataFrame({
    'sentence': textData,
    'cluster': hdbscan_labels_minilm,
    'x': embedding_minilm[:, 0],
    'y': embedding_minilm[:, 1]
})

cluster_bert = pd.DataFrame({
    'sentence': textData,
    'cluster': hdbscan_labels_bert,
    'x': embedding_bert[:, 0],
    'y': embedding_bert[:, 1]
})

# ------------------------- Plotting Clusters ----------------------------

# Creating visualization
fig_tfidf = px.scatter(cluster_tfidf, x='x', y='y', color='cluster', hover_data=['sentence'],
                 title='TF-IDF Tokenized UMAP Projection with HDBSCAN Clustering')
                 
fig_minilm = px.scatter(cluster_minilm, x='x', y='y', color='cluster', hover_data=['sentence'],
                 title='all-MiniLM-L6-v2 Embedded UMAP Projection with HDBSCAN Clustering')
                 
fig_bert = px.scatter(cluster_bert, x='x', y='y', color='cluster', hover_data=['sentence'],
                 title='DistilBERT base model (uncased) Embedded UMAP Projection with HDBSCAN Clustering') 

# Plotting Visualizations
fig_tfidf.show()
fig_minilm.show()
fig_bert.show()
