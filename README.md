# bangla-sms-classicfication
This code appears to be a Python script for performing sentiment analysis on a dataset of Bangla SMS messages. Let's break down what each part of the code does:

Imports: The script imports necessary libraries such as pandas, nltk, re, seaborn, and various classifiers from scikit-learn for machine learning tasks.

Data Loading and Exploration: It loads a CSV file named 'bangla_sms_dataset.csv' into a pandas DataFrame and performs basic exploratory data analysis to understand the structure and contents of the data.

Data Cleaning: There's a function defined (clean_sentence) for cleaning the text data by removing special characters, URLs, HTML tags, and stop words. This function is then applied to the text data column in the DataFrame.

Text Transformation: Another function (text_transformation) is defined for transforming text data by converting all text to lowercase and removing non-alphabetic characters. This transformation is applied to prepare the text data for vectorization.

Feature Extraction: The text data is transformed into numerical features using CountVectorizer from scikit-learn. This step converts the text data into a matrix of token counts.

Model Training: Several machine learning models such as Logistic Regression, Naive Bayes, Support Vector Machines, Decision Trees, XGBoost, Random Forest, Bagging Classifier, and AdaBoost are trained on the feature matrix to classify the SMS messages into different types.

Model Evaluation: The trained models are evaluated using various metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

Visualization: Matplotlib and Seaborn libraries are used to visualize the confusion matrix.

Final Evaluation: The precision, recall, and F1-score are printed out for the final evaluation of the model.

Overall, this script covers the entire pipeline of text preprocessing, feature extraction, model training, evaluation, and visualization for sentiment analysis on Bangla SMS messages.
