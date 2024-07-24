# Clustering and Classifying Misconduct Allegations in Louisiana

## Clustering

The clustering code uses TF-IDF, all-MiniLM-L6-v2, and DistilBERT base model (uncased) embeddings, Uniform Manifold Approximation and Projection (UMAP) dimensionality reduction, and Hierarchical Density-Based Spatial Clustering of Applications with Noise (HBSCAN) clustering. Please see the code for specific parameters.

<img width="1329" alt="Screenshot 2024-07-22 at 5 40 15 PM" src="https://github.com/user-attachments/assets/6dcb5b81-796b-4da7-af6f-92b57cd91e30">

## Classifying

For the classification we use TF-IDF for tokenizing the text. We then fit a Support Vector Classifier for multi-label classification using a 80-20 split and random search cross validation with 10 folds. 

+-------------------------------------+
|       Train      |       Test       |
+------------------+------------------+
|       0.92       |       0.48       |
+------------------+------------------+
|0.9732078204199855|0.7607361963190185|
+------------------+------------------+
|0.9545454545454546|0.6813186813186813|
+------------------+------------------+
|0.9926144756277696|0.8611111111111112|
+------------------+------------------+
|0.9771048969827166|0.8379697225851072|
+-------------------------------------+
