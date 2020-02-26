import gensim
import nltk
import sklearn
import pandas as pd
import numpy as np
import matplotlib

import re
import codecs
import itertools
import matplotlib.pyplot as plt

print ('DONE [IMPORT NECESSARY LIBRARIES]')

input_file = codecs.open("./train.csv", "r",encoding='utf-8', errors='replace')
questions = pd.read_csv(input_file)
#print questions
def standardize_text(df, text_field):
    # normalize by turning all letters into lowercase
    df[text_field] = df[text_field].str.lower()
    # get rid of URLS
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"http\S+", "", elem))  
    return df

# call the text cleaning function
clean_questions = standardize_text(questions, "text")
a = clean_questions.groupby("class_label").count()
#print a

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
clean_questions["token"] = clean_questions["text"].apply(tokenizer.tokenize)
#print clean_questions.head()
all_words = [word for tokens in clean_questions["token"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in clean_questions["token"]]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))

from collections import Counter
count_all_words = Counter(all_words)

# get the top 100 most common occuring words
print count_all_words.most_common(100)

from sklearn.model_selection import train_test_split

list_corpus = clean_questions["text"]
list_labels = clean_questions["class_label"]

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=40)

print("Training set: %d samples" % len(X_train))
print("Test set: %d samples" % len(X_test))

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w+')

bow = dict()
bow["train"] = (count_vectorizer.fit_transform(X_train), y_train)
bow["test"]  = (count_vectorizer.transform(X_test), y_test)
print(bow["train"][0].shape)
print(bow["test"][0].shape)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w+')

tfidf = dict()
tfidf["train"] = (tfidf_vectorizer.fit_transform(X_train), y_train)
tfidf["test"]  = (tfidf_vectorizer.transform(X_test), y_test)

print(tfidf["train"][0].shape)
print(tfidf["test"][0].shape)

word2vec_path = "./GoogleNews-vectors-negative300.bin.gz"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

print ('DONE [Load Word2Vec Pretrained Corpus]')

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions_tokens, generate_missing=False):
    embeddings = clean_questions_tokens.apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)

# Call the functions
embeddings = get_word2vec_embeddings(word2vec, clean_questions['tokens'])

print ('[EMBEDDING] Get Word2Vec values for a Tweet')

X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(embeddings, list_labels, 
                                                                    test_size=0.2, random_state=40)

w2v = dict()
w2v["train"] = (X_train_w2v, y_train_w2v)
w2v["test"]  = (X_test_w2v, y_test_w2v)
print ('DONE - [CLASSIFY] Word2Vec Train Test Split]')

from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', random_state=40)
print ('DONE - [CLASSIFY] Initialize Logistic Regression')

from sklearn.svm import LinearSVC
lsvm_classifier = LinearSVC(C=1.0, class_weight='balanced', multi_class='ovr', random_state=40)
print ('[CLASSIFY] Initialize Support Vector Machine Classifier')

from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()
print ('DONE - [CLASSIFY] Initialize Naive Bayes')

from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
print ('DONE - [CLASSIFY] Initialize Decision Tree')

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1
print ('DONE - [EVALUATE] Prepare Metrics')

from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    return plt
print ('DONE - [EVALUATE] Confusion Matrix')


embedding = bow                  # bow | tfidf | w2v
print ('DONE - [EMBEDDING] CHOOSE EMBEDDING')

classifier = lr_classifier     # lr_classifier | lsvm_classifier | nb_classifier| dt_classifier
print ('DONE - [CLASSIFY] CHOOSE CLASSIFIER')

classifier.fit(*embedding["train"])
y_predict = classifier.predict(embedding["test"][0])

print ('DONE - [CLASSIFY] Train Classifier on Embeddings')

accuracy, precision, recall, f1 = get_metrics(embedding["test"][1], y_predict)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

test_X = pd.read_csv('./test.csv')
test_corpus = test_X["Tweet"]
test_Id = test_X["Id"]
print ('DONE [ETL] Load competition Test Data')

test_corpus_tokens = test_corpus.apply(tokenizer.tokenize)
print ('[PREPROCESS] Tokenize Competition Data')

vectorized_text = dict()
vectorized_text['test']  = (count_vectorizer.transform(test_corpus))  # see options in the above cell
print ('DONE - [EMBEDDING] Apply Chosen Embeddings to the Tweets')

embedding = vectorized_text                
classifier = lr_classifier     # lr_classifier | lsvm_classifier | nb_classifier | dt_classifier
predicted_sentiment = classifier.predict(embedding['test']).tolist()

print ('DONE - [CLASSIFY] Apply Chosen Classifier to the Embedding')

results = pd.DataFrame(
    {'Id': test_Id,
     'Expected': predicted_sentiment
    })
# Write your results for submission.
# Make sure to put in a meaningful name for the 'for_submission.csv 
# to distinguish your submission from other teams.
results.to_csv('for_submission_sample.csv', index=False)
print ('DONE - [PREPARE SUBMISSION]')
