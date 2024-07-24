# Clustering and Classifying Misconduct Allegations in Louisiana

## Purpose
Categorizing police misconduct is difficult. Our allegation data the `allegation` (the allegation) and `allegation_desc` (the allegation description, though infrequently used) columns of `data_allegation.csv`, includes 39,885 allegations of misconduct across hundreds of agencies in Louisiana. Out of these allegations, we downloaded a random sample of 500 unique allegation and allegation description pairs and hand categorized them into 39 labels, allowing for a single allegation-allegation description pair to have multiple labels. 

These categories include: 'Abuse of Authority', 'Adherence to Law', 'Appearance and Cleanliness Violation', 'Arrest Violation', 'Arrest or Conviction', 'Associations Violation', 'Conduct Violation', 'Confidentiality Violation', 'Cooperation Violation', 'Detainment Violation', 'Discourtesy', 'Discrimination', 'Domestic Violence', 'Equipment Misuse and Damage', 'Handling Evidence Violation', 'Harassment', 'In Service Deficiency', 'Insubordination', 'Intimidation and Retaliation', 'Investigation Violation', 'Neglect of Duty', 'Performance of Duty', 'Policy and Procedure Violation', 'Prison-related Violation', 'Professionalism', 'Reporting Violation', 'Search Violation', 'Seizure Violation', 'Sexual Assault', 'Sexual Harassment', 'Stop Violation', 'Substance Violation', 'Supervision Violation', 'Technology Violation', 'Theft', 'Traffic Violation', 'Truthfulness', 'Use of Force', 'Weapon Violation', and 'Miscellaneous Allegation'. 

Here is an example of what three rows:
allegation | allegation_desc | classification|
--- | --- | --- |
rule 2: moral conduct; paragraph 01 - adherence to law|r.s 14:37.7 relative to domestic abuse assault|Adherence to Law, Domestic Violence|
2:14: failure to secure property or evidence|      nan      |Handling Evidence Violation|
dr.03:09.932.01 - failure to follow supervisor instructions, perform work or otherwise comply with policy and procedure - 8 hour suspension|      nan      |Insubordination, Policy and Procedure Violation|

**Note: These classifications are not final**

Specifically, below we describe how we may use clustering algorithms to determine potential new categorizations of misconduct allegations that are less tied to the human classifier's biases. We also describe how we use a Support Vector Classifier to classify the allegations into the 39 categories above.

## Clustering

The purpose of clustering is to give us latent connections between allegation and allegation description pairs that a human classifier may miss. For example, we may be overly focused on grouping by Use of Force, whereas the cluster may do a better job at classifying for all types of physical violence. 

The process of our clustering is as such:
1. Clean all allegation and allegation descriptions from `data_allegation.csv` and concatinate them into a single variable.
2. Fit three embeddings on the single variable: TF-IDF with a maximum of 5000 features, all-MiniLM-L6-v2, and DistilBERT base model (uncased).
3. Fit a Uniform Manifold Approximation and Projection (UMAP) model with 15 nearest neighbors, a minimum distance of 0.1, 2 components, and a cosine metric on each of the three embeddings for dimensionality reduction.
4. Hierarchical Density-Based Spatial Clustering of Applications with Noise (HBSCAN) model with a minimum cluster size of 20 and a euclidean distance metric on each of the dimensionally reduced embeddings to cluster the text. 

**Note: A limitation of our cluster is that it only allows for single-label classification**

Below is an image of the cluster utilizing the BERT embedding:

<img width="1329" alt="Screenshot 2024-07-22 at 5 40 15 PM" src="https://github.com/user-attachments/assets/6dcb5b81-796b-4da7-af6f-92b57cd91e30">

## Classifying

For the classification we use TF-IDF for tokenizing the text. We then fit a Support Vector Classifier for multi-label classification using a 80-20 split and random search cross validation with 10 folds. The best multi-label classifier had C=4.27022004702574 and linear kernel.

The overall metrics of the model include: 

Metric | Train | Test |
--- | --- | --- |
Accuracy | 0.92 | 0.48 |
F1 | 0.97 | 0.76 |
Recall | 0.95 | 0.68 |
Precision | 0.99 | 0.86 |
AUC | 0.98 | 0.84 |

**Note: the accuracy is quite low.**

The variable-specific metrics of the model include:

Label | Test Precision | Test Recall | Test Accuracy | Test TP | Test TN | Test FP | Test FN | Test TP Rate | Test TN Rate | Test FP Rate | Test FN Rate |
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Abuse of Authority | 1.0 | 0.25 | 0.97 | 1 | 96 | 0 | 3 | 0.25 | 1.0 | 0.0 | 0.75 |
Adherence to Law | 1.0 | 1.0 | 1.0 | 2 | 98 | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 |
Appearance and Cleanliness Violation | 0.0 | 0.0 | 1.0 | 100 | 0 | 0 | 0 | 1.0 | 0.0 | 0.0 | 0.0 |
Arrest Violation | 1.0 | 1.0 | 1.0 | 1 | 99 | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 |
Arrest or Conviction | 0.0 | 0.0 | 1.0 | 100 | 0 | 0 | 0 | 1.0 | 0.0 | 0.0 | 0.0 |
Associations Violation | 1.0 | 1.0 | 1.0 | 2 | 98 | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 |
Conduct Violation | 0.88 | 0.70 | 0.96 | 7 | 89 | 1 | 3 | 0.70 | 0.99 | 0.01 | 0.30 |
Confidentiality Violation | 0.0 | 0.0 | 0.99 | 0 | 99 | 0 | 1 | 0.0 | 1.0 | 0.0 | 1.0 |
Cooperation Violation | 0.0 | 0.0 | 0.96 | 0 | 96 | 4 | 0 | 0.0 | 0.96 | 0.04 | 0.0 |
Detainment Violation | 0.0 | 0.0 | 1.0 | 100 | 0 | 0 | 0 | 1.0 | 0.0 | 0.0 | 0.0 |
Discourtesy | 0.86 | 0.67 | 0.96 | 6 | 90 | 1 | 3 | 0.67 | 0.99 | 0.01 | 0.33 |
Discrimination | 0.0 | 0.0 | 0.98 | 0 | 98 | 0 | 2 | 0.0 | 1.0 | 0.0 | 1.0 |
Domestic Violence | 1.0 | 1.0 | 1.0 | 2 | 98 | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 |
Equipment Misuse and Damage | 0.83 | 0.71 | 0.97 | 5 | 92 | 1 | 2 | 0.71 | 0.99 | 0.01 | 0.29 |
Handling Evidence Violation | 0.0 | 0.0 | 0.98 | 0 | 98 | 0 | 2 | 0.0 | 1.0 | 0.0 | 1.0 |
Harassment | 0.0 | 0.0 | 0.98 | 0 | 98 | 0 | 2 | 0.0 | 1.0 | 0.0 | 1.0 |
In Service Deficiency | 0.0 | 0.0 | 0.99 | 0 | 99 | 1 | 0 | 0.0 | 0.99 | 0.01 | 0.0 |
Insubordination | 0.88 | 0.88 | 0.96 | 14 | 82 | 2 | 2 | 0.88 | 0.98 | 0.02 | 0.12 |
Intimidation and Retaliation | 1.0 | 0.5 | 0.98 | 2 | 96 | 0 | 2 | 0.5 | 1.0 | 0.0 | 0.5 |
Investigation Violation | 0.0 | 0.0 | 0.99 | 0 | 99 | 0 | 1 | 0.0 | 1.0 | 0.0 | 1.0 |
Neglect of Duty | 0.84 | 0.91 | 0.91 | 31 | 60 | 6 | 3 | 0.91 | 0.91 | 0.09 | 0.09 |
Performance of Duty | 1.0 | 0.91 | 0.98 | 21 | 77 | 0 | 2 | 0.91 | 1.0 | 0.0 | 0.09 |
Policy and Procedure Violation | 1.0 | 0.5 | 0.97 | 3 | 94 | 0 | 3 | 0.5 | 1.0 | 0.0 | 0.5 |
Prison-related Violation | 1.0 | 0.44 | 0.95 | 4 | 91 | 0 | 5 | 0.44 | 1.0 | 0.0 | 0.56 |
Professionalism | 1.0 | 0.67 | 0.97 | 6 | 91 | 0 | 3 | 0.67 | 1.0 | 0.0 | 0.33 |
Reporting Violation | 0.0 | 0.0 | 0.99 | 0 | 99 | 0 | 1 | 0.0 | 1.0 | 0.0 | 1.0 |
Search Violation | 0.0 | 0.0 | 0.98 | 0 | 98 | 0 | 2 | 0.0 | 1.0 | 0.0 | 1.0 |
Seizure Violation | 0.0 | 0.0 | 1.0 | 100 | 0 | 0 | 0 | 1.0 | 0.0 | 0.0 | 0.0 |
Sexual Assault | 1.0 | 1.0 | 1.0 | 1 | 99 | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 |
Sexual Harassment | 1.0 | 0.5 | 0.99 | 1 | 98 | 0 | 1 | 0.5 | 1.0 | 0.0 | 0.5 |
Stop Violation | 0.0 | 0.0 | 0.98 | 0 | 98 | 0 | 2 | 0.0 | 1.0 | 0.0 | 1.0 |
Substance Violation | 0.0 | 0.0 | 0.99 | 0 | 99 | 0 | 1 | 0.0 | 1.0 | 0.0 | 1.0 |
Supervision Violation | 1.0 | 0.5 | 0.99 | 1 | 98 | 0 | 1 | 0.5 | 1.0 | 0.0 | 0.5 |
Technology Violation | 0.86 | 0.67 | 0.96 | 6 | 90 | 1 | 3 | 0.67 | 0.99 | 0.01 | 0.33 |
Theft | 0.0 | 0.0 | 0.99 | 0 | 99 | 0 | 1 | 0.0 | 1.0 | 0.0 | 1.0 |
Traffic Violation | 0.0 | 0.0 | 0.99 | 0 | 99 | 1 | 0 | 0.0 | 0.99 | 0.01 | 0.0 |
Truthfulness | 0.50 | 0.25 | 0.96 | 1 | 95 | 1 | 3 | 0.25 | 0.99 | 0.01 | 0.75 |
Use of Force | 0.86 | 0.67 | 0.96 | 6 | 90 | 1 | 3 | 0.67 | 0.99 | 0.01 | 0.33 |
Weapon Violation | 0.0 | 0.0 | 0.98 | 0 | 98 | 0 | 2 | 0.0 | 1.0 | 0.0 | 1.0 |
