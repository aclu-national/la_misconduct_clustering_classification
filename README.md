# Clustering and Classifying Misconduct Allegations in Louisiana

## Clustering

The clustering code uses TF-IDF, all-MiniLM-L6-v2, and DistilBERT base model (uncased) embeddings, Uniform Manifold Approximation and Projection (UMAP) dimensionality reduction, and Hierarchical Density-Based Spatial Clustering of Applications with Noise (HBSCAN) clustering. Please see the code for specific parameters.

<img width="1329" alt="Screenshot 2024-07-22 at 5 40 15 PM" src="https://github.com/user-attachments/assets/6dcb5b81-796b-4da7-af6f-92b57cd91e30">

## Classifying

For the classification we use TF-IDF for tokenizing the text. We then fit a Support Vector Classifier for multi-label classification using a 80-20 split and random search cross validation with 10 folds. 

Attempt | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Seconds | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
