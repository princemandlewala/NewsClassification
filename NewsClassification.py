
# coding: utf-8

# # Import packages

# In[140]:

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from textblob.classifiers import NaiveBayesClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm
from nltk.corpus import stopwords
from tabulate import tabulate
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.pipeline import Pipeline


# # Utility functions

# In[117]:

def print_correct_percentage(confusionMatrix):
    matrix = np.matrix(confusionMatrix)
    sum = 0
    for i in range(0,5):
        sum = sum + confusionMatrix[i,i]
    return round(sum/matrix.sum()*100,2)


# ### Classification of news using Multinomial Naive Bayes Classifier

# ### 1). Without removing stop words

# In[118]:

news = pd.read_csv('D:/datasets/news_classification/dataset.csv', encoding='cp1252')
cv = CountVectorizer()
newsTrainCv = cv.fit_transform(news.news)
tf_transformer = TfidfTransformer(use_idf=True).fit(newsTrainCv)
newsTrainCv1 = tf_transformer.transform(newsTrainCv)
clf = MultinomialNB(alpha=0.003)
news_train, news_test,type_train, type_test = train_test_split(newsTrainCv1,news.type,test_size = 0.2, random_state=20)
news_trained = clf.fit(news_train,type_train)
news_predicted = news_trained.predict(news_test)
confusionMatrix = confusion_matrix(type_test,news_predicted)
percentage_NB = print_correct_percentage(confusionMatrix)
print("Percentage of news classified correctly using Multinomial Naive Bayes Classifier: ", percentage_NB,"%")
print(classification_report(type_test,news_predicted))


# ### 2). After removing stop words

# In[119]:

news = pd.read_csv('D:/datasets/news_classification/dataset.csv', encoding='cp1252')
stop = set(stopwords.words('english'))
list = []
for data in news.news:
    words = data.split()
    s = ''
    for w in words:
        if w not in stop:
            s = s + w + " "
    list.append(s)
newsDataFrame = pd.DataFrame(list,columns=['news'])
cv = CountVectorizer()
newsTrainCv = cv.fit_transform(newsDataFrame.news)
tf_transformer = TfidfTransformer(use_idf=True).fit(newsTrainCv)
newsTrainCv1StopWords = tf_transformer.transform(newsTrainCv)
clf = MultinomialNB(alpha=0.003)
news_train, news_test,type_train, type_test = train_test_split(newsTrainCv1StopWords,news.type,test_size = 0.2, random_state=20)
news_trained = clf.fit(news_train,type_train)
news_predicted = news_trained.predict(news_test)
confusionMatrix = confusion_matrix(type_test,news_predicted)
percentage_NB = print_correct_percentage(confusionMatrix)
print("Percentage of news classified correctly using Multinomial Naive Bayes Classifier: ", percentage_NB,"%")
print(classification_report(type_test,news_predicted))


# ### Classification of news using SVM

# ### 1). Without removing stop words

# In[120]:

clfSVC = svm.LinearSVC(C=0.5)
news_train_svc, news_test_svc,type_train_svc, type_test_svc = train_test_split(newsTrainCv1,news.type,test_size = 0.2, random_state=40)
news_trained_svc = clfSVC.fit(news_train_svc, type_train_svc)
news_predicted_svc = news_trained_svc.predict(news_test_svc)
confusionMatrixSVC = confusion_matrix(type_test_svc,news_predicted_svc)
percentage_SVC = print_correct_percentage(confusionMatrixSVC)
print("Percentage of news classified correctly using Support Vector Classifier: ", percentage_SVC,"%")
print(classification_report(type_test_svc,news_predicted_svc))


# ### 2). After removing stop words

# In[121]:

clfSVC = svm.LinearSVC(C=0.5)
news_train_svc, news_test_svc,type_train_svc, type_test_svc = train_test_split(newsTrainCv1StopWords,news.type,test_size = 0.2, random_state=40)
news_trained_svc = clfSVC.fit(news_train_svc, type_train_svc)
news_predicted_svc = news_trained_svc.predict(news_test_svc)
confusionMatrixSVC = confusion_matrix(type_test_svc,news_predicted_svc)
percentage_SVC = print_correct_percentage(confusionMatrixSVC)
print("Percentage of news classified correctly using Support Vector Classifier: ", percentage_SVC,"%")
print(classification_report(type_test_svc,news_predicted_svc))

