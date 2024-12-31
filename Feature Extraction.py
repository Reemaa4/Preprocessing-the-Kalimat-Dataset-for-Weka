import arff
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
import pandas as pd

# Define path to ARFF file
arff_file_path = r'C:\Users\Lapto\OneDrive\سطح المكتب\NLP assignment\ar_text_dataset.arff'

def process_files_for_arff(arff_file_path):
    # Load ARFF file
    with open(arff_file_path, 'r', encoding='utf-8') as f:
        dataset = arff.load(f)
    
    # Extract sentences and labels
    all_sentences = []
    all_labels = []
    for entry in dataset['data']:
        sentence = entry[0]  # Assuming the text is in the first column
        label = entry[1]     # Assuming the label is in the second column
        all_sentences.append(sentence)
        all_labels.append(label)
    
    return all_sentences, all_labels

# Load the ARFF file and preprocess the data
all_sentences, all_labels = process_files_for_arff(arff_file_path)

# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(all_sentences, all_labels, test_size=0.2, random_state=42)

# Vectorize the text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Limit the number of features to 1000
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Adding two handcrafted features (e.g., text length, average word length)
train_text_lengths = np.array([len(text) for text in X_train]).reshape(-1, 1)
train_avg_word_lengths = np.array([np.mean([len(word) for word in text.split()]) for text in X_train]).reshape(-1, 1)

test_text_lengths = np.array([len(text) for text in X_test]).reshape(-1, 1)
test_avg_word_lengths = np.array([np.mean([len(word) for word in text.split()]) for text in X_test]).reshape(-1, 1)

# Concatenate handcrafted features with TF-IDF features
X_train_combined = np.hstack((X_train_tfidf.toarray(), train_text_lengths, train_avg_word_lengths))
X_test_combined = np.hstack((X_test_tfidf.toarray(), test_text_lengths, test_avg_word_lengths))

# Apply feature selection to select top 20 features
selector = SelectKBest(chi2, k=20)
X_train_selected = selector.fit_transform(X_train_combined, y_train)
X_test_selected = selector.transform(X_test_combined)

# Convert the selected features back to a pandas DataFrame for further use
columns = ['feature_' + str(i) for i in range(X_train_selected.shape[1])]
train_df = pd.DataFrame(X_train_selected, columns=columns)
test_df = pd.DataFrame(X_test_selected, columns=columns)

# Save the selected features as ARFF
def save_arff(df, labels, filename):
    arff_data = {
        'description': '',
        'relation': 'text_classification',
        'attributes': [(f'feature_{i}', 'REAL') for i in range(df.shape[1])] + [('class', list(set(labels)))],
        'data': [list(row) + [label] for row, label in zip(df.values, labels)]
    }
    with open(filename, 'w', encoding='utf-8') as f:
        arff.dump(arff_data, f)

# Save training and test ARFF files with selected features
save_arff(train_df, y_train, 'train_selected.arff')
save_arff(test_df, y_test, 'test_selected.arff')

print("ARFF files with selected features have been saved successfully.")
# =====================================================================================
# import arff
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_selection import SelectKBest, chi2
# import pandas as pd

# # Define path to ARFF file
# arff_file_path = r'C:\Users\Lapto\OneDrive\سطح المكتب\NLP assignment\example_articles.arff'

# def process_files_for_arff(arff_file_path):
#     # Load ARFF file
#     with open(arff_file_path, 'r', encoding='utf-8') as f:
#         dataset = arff.load(f)
    
#     # Extract sentences and labels
#     all_sentences = []
#     all_labels = []
#     for entry in dataset['data']:
#         sentence = entry[0]  # Assuming the text is in the first column
#         label = entry[1]     # Assuming the label is in the second column
#         all_sentences.append(sentence)
#         all_labels.append(label)
    
#     return all_sentences, all_labels

# # Load the ARFF file and preprocess the data
# all_sentences, all_labels = process_files_for_arff(arff_file_path)

# # Vectorize the text using TF-IDF
# tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Limit the number of features to 1000
# X_tfidf = tfidf_vectorizer.fit_transform(all_sentences)

# # Adding two handcrafted features (e.g., text length, average word length)
# text_lengths = np.array([len(text) for text in all_sentences]).reshape(-1, 1)
# avg_word_lengths = np.array([np.mean([len(word) for word in text.split()]) for text in all_sentences]).reshape(-1, 1)

# # Concatenate handcrafted features with TF-IDF features
# X_combined = np.hstack((X_tfidf.toarray(), text_lengths, avg_word_lengths))

# # Apply feature selection to select top 20 features
# selector = SelectKBest(chi2, k=20)
# X_selected = selector.fit_transform(X_combined, all_labels)

# # Convert the selected features back to a pandas DataFrame for further use
# columns = ['feature_' + str(i) for i in range(X_selected.shape[1])]
# df_selected = pd.DataFrame(X_selected, columns=columns)

# # Save the selected features as ARFF
# def save_arff(df, labels, filename):
#     arff_data = {
#         'description': '',
#         'relation': 'text_classification',
#         'attributes': [(f'feature_{i}', 'REAL') for i in range(df.shape[1])] + [('class', list(set(labels)))],
#         'data': [list(row) + [label] for row, label in zip(df.values, labels)]
#     }
#     with open(filename, 'w', encoding='utf-8') as f:
#         arff.dump(arff_data, f)

# # Save the ARFF file with the selected features
# save_arff(df_selected, all_labels, 'selected_features.arff')

# print("ARFF file with selected features has been saved successfully.")
