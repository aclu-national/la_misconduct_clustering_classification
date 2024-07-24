'''
Code to classify misconduct allegations using a Support Vector Classifier and multi-label classification.
Using random search cross-validation with 10 folds.
'''

# -----------------------------------------------------------------------
# Author: Elijah Appelson
# Update Date: July 24th, 2024
# -----------------------------------------------------------------------

# Importing Packages
import pandas as pd
import re
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, multilabel_confusion_matrix
from scipy.stats import uniform
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer,AutoModel
from sentence_transformers import SentenceTransformer


# ------------------------ Defining Text Cleaning -----------------------------

# Defining stop words, lemmatization, and preprocess functions
def remove_stopwords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def preprocess(text):
    text = str(text.split(':')[-1])
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\d_]+', '', text)
    text = remove_stopwords(text)
    return lemmatize_text(text)

# --------------------- Loading and Cleaning Data -----------------------------

# Reading the data
df = pd.read_csv("labelled_data.csv")

# Cleaning the allegation and allegation descriptions
df['allegation_clean'] = df['allegation'].dropna().apply(preprocess)
df['description_clean'] = df['allegation_desc'].dropna().apply(preprocess)

# Combine the cleaned allegation and description into one
df['combined'] = df['allegation_clean'].fillna('') + " " + df['description_clean'].fillna('')

# Turning the classifications into a list of classifications for multi-label
df['combined_list'] = df['classification'].apply(lambda x: [item.strip() for item in x.split(',')])

# Creating the MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Fitting the MultiLabelBinarizer on our classification lists
binary_matrix = mlb.fit_transform(df['combined_list'])

# Creating a dataframe out of the MultiLabelBinarizer
binary_df = pd.DataFrame(binary_matrix, columns=mlb.classes_)

# Removing the Miscellaneous Allegation variable
if "Miscellaneous Allegation" in binary_df.columns:
    binary_df = binary_df.drop(columns=["Miscellaneous Allegation"])

# Concatenating our dataframe with binary_df
df = pd.concat([df[['combined']], binary_df], axis=1)

# ------------------- Splitting into train-test -----------------------------

# Split the data into training and test sets with an 80-20 split
test_split = 0.2
X = df[['combined']].values
y = df.drop(['combined'], axis=1).values
X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=test_split)

# Converting back to DataFrame for consistency
train_df = X_train.flatten()
train_labels = y_train
test_df =  X_test.flatten()
test_labels = y_test

# ----------------- Creating Support Vector Classifier ------------------------

# Three different tokenizations / embeddings
tfidf = TfidfVectorizer(max_features=2000)
minilm = SentenceTransformer('all-MiniLM-L6-v2')
bert = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# Defining the Support Vector Classifier
svc = SVC(probability=True)

# Defining the possible parameters
param_dist = {
    'classifier__estimator__C': uniform(0.1, 10),  
    'classifier__estimator__kernel': ['linear', 'rbf','poly'], 
    'classifier__estimator__gamma': ['scale', 'auto']
}

# Defining the MultiOutputClassifier
multi_target_svc = MultiOutputClassifier(svc)

# Creating our pipeline
pipeline = Pipeline([
    ('vectorizer', minilm),
    ('classifier', multi_target_svc)
])

# Running Random Search Cross Validation
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=10,
    verbose=1,
    random_state=1,
    n_jobs=-1
)

# Fitting the random search cross validation
random_search.fit(train_df, train_labels)

# Pulling the best model
pipeline = random_search.best_estimator_

# ----------------- Creating Random Forest Classifier ------------------------

# ----------------- Creating XG Boost Classifier ------------------------

# ----------------- Creating RNN Classifier ------------------------

# ----------------- Creating BERT Classifier ------------------------

# ------------------------ Broad Metrics --------------------------

# Predicting
train_preds = pipeline.predict(train_df)
test_preds = pipeline.predict(test_df)

# Counting how many predictions are all 0's 
np.sum(np.all(test_preds == 0, axis=1))

# Evaluating the model
train_accuracy = accuracy_score(train_labels, train_preds)
test_accuracy = accuracy_score(test_labels, test_preds)
train_f1 = f1_score(train_labels, train_preds, average='micro')
test_f1 = f1_score(test_labels, test_preds, average='micro')
train_recall = recall_score(train_labels, train_preds, average = "micro")
test_recall = recall_score(test_labels, test_preds, average = "micro")
train_precision = precision_score(train_labels, train_preds, average = "micro")
test_precision = precision_score(test_labels, test_preds, average = "micro")
train_auc = roc_auc_score(train_labels, train_preds, average = "micro")
test_auc = roc_auc_score(test_labels, test_preds, average = "micro")

# Turning metrics into a dataframe
data = {
    'Train': [train_accuracy, train_f1, train_recall, train_precision, train_auc],
    'Test': [test_accuracy, test_f1, test_recall, test_precision, test_auc]
}

metrics = ['Accuracy', 'F1', 'Recall', 'Precision', 'AUC']

# Overall Metrics Dataframe
overall_metrics = pd.DataFrame(data, index=metrics)

# ---------------------- Variable-Specific Metrics --------------------------

label_columns = binary_df.columns

# Defining the metrics_data list
metrics_data = []

for i, label in enumerate(label_columns):
    # Defining test metrics
    test_precision = precision_score(test_labels[:, i], test_preds[:, i], average='binary', zero_division=0)
    test_recall = recall_score(test_labels[:, i], test_preds[:, i], average='binary', zero_division=0)
    test_accuracy = accuracy_score(test_labels[:, i], test_preds[:, i])
    test_confusion = confusion_matrix(test_labels[:, i], test_preds[:, i])
    
    def extract_metrics(conf_matrix):
        tn = fp = fn = tp = 0
        if conf_matrix.shape == (2, 2):
            tn, fp, fn, tp = conf_matrix.ravel()
        elif conf_matrix.shape == (1, 2):  # Only negative cases in ground truth
            tn, fp = conf_matrix.ravel()
        elif conf_matrix.shape == (2, 1):  # Only positive cases in ground truth
            fn, tp = conf_matrix.ravel()
        elif conf_matrix.shape == (1, 1):  # Only one type in both prediction and ground truth
            if len(set(test_labels[:, i])) == 1 and len(set(test_preds[:, i])) == 1:
                if test_labels[0, i] == 0:
                    tn = conf_matrix[0, 0]
                else:
                    tp = conf_matrix[0, 0]
        else:
            raise ValueError("Unknown Error")
        return tn, fp, fn, tp
    
    test_tn, test_fp, test_fn, test_tp = extract_metrics(test_confusion)
    
    # Calculating the tp, fp, tn, and fn rates
    test_tp_rate = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
    test_tn_rate = test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0
    test_fp_rate = test_fp / (test_fp + test_tn) if (test_fp + test_tn) > 0 else 0
    test_fn_rate = test_fn / (test_fn + test_tp) if (test_fn + test_tp) > 0 else 0
    
    # Appending all rates to the metrics_data list
    metrics_data.append({
        'Label': label,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test Accuracy': test_accuracy,
        'Test TP': test_tp,
        'Test TN': test_tn,
        'Test FP': test_fp,
        'Test FN': test_fn,
        'Test TP Rate': test_tp_rate,
        'Test TN Rate': test_tn_rate,
        'Test FP Rate': test_fp_rate,
        'Test FN Rate': test_fn_rate
    })

# Converting the list into a dataframe of variable metrics
variable_metrics = pd.DataFrame(metrics_data)

# ---------------------- Using the model --------------------------

# Creating a list of classes without "Miscellaneous Allegation"
filtered_classes = [cls for cls in mlb.classes_ if cls != "Miscellaneous Allegation"]
filtered_classes
# Defining a prediction function take in text and run it though our pipeline to check its classification
def predict(text, pipeline):
    preprocessed_text = preprocess(text)
    features = pipeline.named_steps['vectorizer'].transform([preprocessed_text])
    predictions = pipeline.named_steps['classifier'].predict(features)
    indices = [i for i, value in enumerate(predictions[0]) if value == 1]
    filtered_values = [filtered_classes[i] for i in indices]
    if len(filtered_values) != 0:
      return(filtered_values)
    else:
      return("Miscellaneous Allegation")

# Testing Model on text
predict("Weapon e", pipeline)


