#  SEMANTIC: Systematic Evaluation of Multi-class Article News-Text Identification and Categorization
SEMANTIC is a project focused on automating the multi-class classification of news articles using machine learning. By leveraging both traditional models and advanced NLP techniques like CNNs and BERT, it addresses key challenges such as class imbalance and varied article lengths. Using an IAB-categorized dataset, SEMANTIC aims to improve contextual understanding and accurate categorization of diverse news content.

## Project Overview
This project, SEMANTIC, focuses on the systematic evaluation of multi-class classification models for categorizing news articles into IAB 
(Interactive Advertising Bureau) categories. We explore the use of traditional and deep learning models, emphasizing the challenges of class imbalance and variable-length articles, with a dataset categorized under the IAB taxonomy.

## Key Features
Multi-class Classification: Classify news articles into categories based on IAB standards.
Imbalance Handling: Techniques to address imbalanced classes.
Contextual Embedding Models: Utilization of Word2Vec and BERT for improved semantic understanding.

## Dataset
Semantic leverages a labeled dataset from HuggingFace containing IAB-categorized news articles, which provides a real-world representation of news content across various categories. The dataset contains 26 different categories ranging over a variety of content, like "academic interests", "sports", "video games", and more. The dataset presents challenges due to:

1. **Class Imbalance**: Some categories have significantly fewer articles.
2. **Variable Article Lengths**: Articles range from a few hundred to thousands of characters.

Data is split into Train, Validation, Test in  45%, 35%, 20% respectively.

## Core Task of SEMANTIC
The core task of S.E.M.A.N.T.I.C is multi-class classification of news articles. This involves:
* Preparing the data and accounting for the class imbalance.
* Building a Logistic Regression, CNN, and BERT-based classifier.
* Assigning each article to one or more IAB categories.
* Handling a diverse range of topics and writing styles.


## Model Overview
SEMANTIC compares three models:
### 1. Baseline model - Logistic Regression with TF-IDF:
A simple yet effective approach for text classification.
### 2. Intermediate - CNN with Word2Vec embeddings:
Combines word embeddings with convolutional layers to capture local patterns in text.
### 3. Advanced - BERT:
Utilizes transformer architecture for contextual understanding of text.

### Model Selection - CNN with Word2Vec
* CNN with Word2Vec is chosen as the primary model after evalulating the 3 models.
* This architecture allows the model to effectively capture local patterns in text while considering the semantic relationships between words.
* It shows significant improvement from the initial to final accuracy, indicating effective learning.
