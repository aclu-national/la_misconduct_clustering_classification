# Clustering and Classifying Misconduct Allegations in Louisiana

## Purpose
Categorizing police misconduct is difficult. Our allegation data, the `allegation` (the allegation) and `allegation_desc` (the allegation description, though infrequently used) columns of `data_allegation.csv`, includes 39,885 allegations of misconduct across hundreds of agencies in Louisiana. Out of these allegations, we downloaded a random sample of 500 unique allegation and allegation description pairs and hand categorized them into 39 labels, allowing for a single allegation-allegation description pair to have multiple labels. 

These categories include: 'Abuse of Authority', 'Adherence to Law', 'Appearance and Cleanliness Violation', 'Arrest Violation', 'Arrest or Conviction', 'Associations Violation', 'Conduct Violation', 'Confidentiality Violation', 'Cooperation Violation', 'Detainment Violation', 'Discourtesy', 'Discrimination', 'Domestic Violence', 'Equipment Misuse and Damage', 'Handling Evidence Violation', 'Harassment', 'In Service Deficiency', 'Insubordination', 'Intimidation and Retaliation', 'Investigation Violation', 'Neglect of Duty', 'Performance of Duty', 'Policy and Procedure Violation', 'Prison-related Violation', 'Professionalism', 'Reporting Violation', 'Search Violation', 'Seizure Violation', 'Sexual Assault', 'Sexual Harassment', 'Stop Violation', 'Substance Violation', 'Supervision Violation', 'Technology Violation', 'Theft', 'Traffic Violation', 'Truthfulness', 'Use of Force', 'Weapon Violation', and 'Miscellaneous Allegation'. 

Here is an example of three classified rows:
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

Here is an example of the final clustering for the cleaned allegation and allegation description pair "neglect duty":
cleaned_sentence | cluster | x | y |
--- | --- | ---| ---|
|neglect duty|   13  |2.6262362003326416|1.1222500801086426 |

**Note: A limitation of our cluster is that it only allows for single-label classification**

<img width="1329" alt="Screenshot 2024-07-22 at 5 40 15 PM" src="https://github.com/user-attachments/assets/6dcb5b81-796b-4da7-af6f-92b57cd91e30">

## Classifying

The purpose of the classification is to classify each of our allegation and allegation description pairs into one or multiple of the 39 categories provided above. To build this model we need to recognize certain limitations including that our sample of allegations may, and likely does not, represents all potential categories of misconduct in Louisiana. We also must recognize that the limited training data will likely diminish its predictive accuracy. Finally, it is important to note that the model will reflect the biases of the human classifier in choosing the categories for the sample of 500 allegation and allegation description pairs. 

The process of our classification is as such:
1. Clean all unique allegation and allegation descriptions from `labelled_data.csv` and concatinate them into a single variable.
2. Use `MultiLabelBinarizer` to transform our classification lists into 39 unique variables and combine this with our original single variable allegation and allegation description pairs.
3. Split our data into a stratified 80-20 (400 training, 100 test) train-test split using a random state of 1.
4. Define our Support Vector Classifier, letting probabilities be true, and our MultiOutputClassifier.
5. Creating a model pipeline with TF-IDF tokenization and a maximum of 500 features.
6. Fitting our model on the training data using a randomized search cross validation with 10 iterations and 10 folds along with potentional parameters of a C value between 0.1 and 10, a linear and radial kernal, and gamma as scale and auto.
7. Computing broad and variable-specific model metrics of our model on the test and training data.

**Note: The classification will include TF-IDF, all-MiniLM-L6-v2, and DistilBERT base model (uncased) embeddings along with other model types including a Random Forest classifier, a Bidirectional Reccurent Nueral Network, and a BERT classification model soon**

We found that the best multi-label classifier had `C=4.27022004702574` and linear kernel.

The overall metrics of the model include: 

Metric | Train | Test |
--- | --- | --- |
Accuracy | 0.98 | 0.56 |
Micro Average F1 | 0.99 | 0.82 |
Micro Average Recall | 0.99 | 0.75 |
Micro Average Precision | 1.00 | 0.91 |
Micro Average AUC | 0.99 | 0.87 |

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





Label|Test Precision|Test Recall|Test Accuracy|Test TP|Test TN|Test FP|Test FN|Test TP Rate|Test TN Rate|Test FP Rate|Test FN Rate
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Abuse of Authority|0.0|0.0|0.9803921568627451|0|100|0|2|0.0|1.0|0.0|1.0
Adherence to Law|1.0|1.0|1.0|2|100|0|0|1.0|1.0|0.0|0.0
Appearance and Cleanliness Violation|0.0|0.0|0.9803921568627451|0|100|0|2|0.0|1.0|0.0|1.0
Arrest Violation|1.0|1.0|1.0|2|100|0|0|1.0|1.0|0.0|0.0
Arrest or Conviction|0.0|0.0|0.9901960784313726|0|101|0|1|0.0|1.0|0.0|1.0
Associations Violation|1.0|1.0|1.0|2|100|0|0|1.0|1.0|0.0|0.0
Conduct Violation|0.9|1.0|0.9901960784313726|9|92|1|0|1.0|0.989247311827957|0.010752688172043012|0.0
Confidentiality Violation|0.0|0.0|1.0|0|102|0|0|0.0|1.0|0.0|0.0
Cooperation Violation|0.0|0.0|0.9803921568627451|0|100|0|2|0.0|1.0|0.0|1.0
Detainment Violation|0.0|0.0|1.0|0|102|0|0|0.0|1.0|0.0|0.0
Discourtesy|0.9|0.8181818181818182|0.9705882352941176|9|90|1|2|0.8181818181818182|0.989010989010989|0.01098901098901099|0.18181818181818182
Discrimination|0.0|0.0|0.9803921568627451|0|100|1|1|0.0|0.9900990099009901|0.009900990099009901|1.0
Domestic Violence|1.0|0.5|0.9901960784313726|1|100|0|1|0.5|1.0|0.0|0.5
Equipment Misuse and Damage|1.0|0.5|0.9803921568627451|2|98|0|2|0.5|1.0|0.0|0.5
Handling Evidence Violation|1.0|0.5|0.9901960784313726|1|100|0|1|0.5|1.0|0.0|0.5
Harassment|1.0|1.0|1.0|1|101|0|0|1.0|1.0|0.0|0.0
In Service Deficiency|0.0|0.0|1.0|0|102|0|0|0.0|1.0|0.0|0.0
Insubordination|1.0|0.8421052631578947|0.9705882352941176|16|83|0|3|0.8421052631578947|1.0|0.0|0.15789473684210525
Intimidation and Retaliation|0.6666666666666666|0.6666666666666666|0.9803921568627451|2|98|1|1|0.6666666666666666|0.98989898989899|0.010101010101010102|0.3333333333333333
Investigation Violation|1.0|1.0|1.0|1|101|0|0|1.0|1.0|0.0|0.0
Neglect of Duty|0.9411764705882353|0.8205128205128205|0.9117647058823529|32|61|2|7|0.8205128205128205|0.9682539682539683|0.031746031746031744|0.1794871794871795
Performance of Duty|1.0|1.0|1.0|17|85|0|0|1.0|1.0|0.0|0.0
Policy and Procedure Violation|1.0|0.4|0.9705882352941176|2|97|0|3|0.4|1.0|0.0|0.6
Prison-related Violation|0.8571428571428571|0.75|0.9705882352941176|6|93|1|2|0.75|0.9893617021276596|0.010638297872340425|0.25
Professionalism|0.8|0.5714285714285714|0.9607843137254902|4|94|1|3|0.5714285714285714|0.9894736842105263|0.010526315789473684|0.42857142857142855
Reporting Violation|1.0|0.6666666666666666|0.9901960784313726|2|99|0|1|0.6666666666666666|1.0|0.0|0.3333333333333333
Search Violation|1.0|1.0|1.0|2|100|0|0|1.0|1.0|0.0|0.0
Seizure Violation|1.0|1.0|1.0|1|101|0|0|1.0|1.0|0.0|0.0
Sexual Assault|0.5|1.0|0.9901960784313726|1|100|1|0|1.0|0.9900990099009901|0.009900990099009901|0.0
Sexual Harassment|1.0|1.0|1.0|1|101|0|0|1.0|1.0|0.0|0.0
Stop Violation|0.0|0.0|0.9901960784313726|0|101|1|0|0.0|0.9901960784313726|0.00980392156862745|0.0
Substance Violation|0.0|0.0|0.9803921568627451|0|100|0|2|0.0|1.0|0.0|1.0
Supervision Violation|1.0|1.0|1.0|2|100|0|0|1.0|1.0|0.0|0.0
Technology Violation|1.0|0.625|0.9705882352941176|5|94|0|3|0.625|1.0|0.0|0.375
Theft|0.0|0.0|1.0|0|102|0|0|0.0|1.0|0.0|0.0
Traffic Violation|1.0|0.3333333333333333|0.9803921568627451|1|99|0|2|0.3333333333333333|1.0|0.0|0.6666666666666666
Truthfulness|0.75|0.75|0.9803921568627451|3|97|1|1|0.75|0.9897959183673469|0.01020408163265306|0.25
Use of Force|0.7777777777777778|0.875|0.9705882352941176|7|92|2|1|0.875|0.9787234042553191|0.02127659574468085|0.125
Weapon Violation|0.0|0.0|0.9705882352941176|0|99|1|2|0.0|0.99|0.01|1.0

