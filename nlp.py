import pandas as pd 
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import scikitplot as skplt
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud



#test Data - NLP problem
# stemming and Lamitization 
# count vectorizer / TFIDF
# train test
# Model building
# Check accuracy

class NLP:
    def __inti__(self,data):
        self.data = data

    
    def stemming(self, column_name):
        try:
            curpos = []
            stemming = PorterStemmer()
            for i in range(len(self.data)):
                tweet = re.sub('[a-zA-Z]', " ",self.data[column_name][i])
                tweet = re.sub('http', '', tweet)
                tweet = tweet.lower()
                tweet = tweet.split()
                #tweet = [stemming.stem(word) for word in tweet if word not in set(stopwords.words("english"))]
                tweet = "".join(tweet)
                curpos.append(tweet)
        

        except Exception as e:
            print("Stemming Error : ", e)
        
        else:
            # print("Cleaning was successful")
            return curpos

#Lemmatization

    def lemmatizing(self, column_name):
        """ cleaning data using re, lemmatization and stopwords"""
        try:
            curpos = []
            stemming = WordNetLemmatizer()
            for i in range(len(self.data)):
                tweet = re.sub('[a-zA-Z]', " ",self.data[column_name][i])
                tweet = re.sub('http', '', tweet)
                tweet = tweet.lower()
                tweet = tweet.split()
                tweet = [stemming.lemmatize(word) for word in tweet if word not in set{stopwords.words("english")}]
                tweet = "".join(tweet)
                curpos.append(tweet)
        

        except Exception as e:
            print("lemmatization Error : ", e)

        else:
            print("cleaning was completed")
            return curpos
        

# count vectorizer / TFIDF
    def count_vectorizer(self,curpos, max_features = 3000, ngram_range(1,2)):
        #bag of words
        try:
            cv = CountVectorizer(max_features= max_features, ngram_range = ngram_range)
            x = cv.fit_transform(curpos).toarray()
        except Exception as e :
            print("Count Vectorizer Error", e)

        else:
            # print("Bag of words created successfully")
            return x
        
# TFIDF
    def tf_idf(self, curpos, max_features = 3000, ngram_range(1,2)):
        #bag of words
        try:
            tfidf = TfidfVectorizer(max_features= max_features, ngram_range = ngram_range)
            X = tfidf.fit_tranform(curpos).toarray()
        except Exception as e :
            print("TFIDF Error", e)

        else:
            # print("Bag of words created successfully")
            return X
    
    def y_encoding(self, target_label):
        """ One Hot Encoding if target variable are not in form of 1s and 0s"""
        try:
            y = pd.get_dummies(self.data[target_label], drop_first = True)

        except Exception as e:
            print("y_encoding error : ", e)
        
        else:
            #print("y encoded succcessfully")
            return y
        
    
    def spilt_data(self, X, y, test_size =0.25, random_state = 0):
        """ Splitting data into train test set"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
        
        except Exception as e:
            print("split_data Error :", e)

        else:
            # print("Successful Splitting")
            return X_train, X_test, y_train, y_test
        
    
    def naive_model(self, X_train, X_test, y_train, y_test):
        """Prediction of mof=del using naive_bayes"""
        try:
            naive = MultinomialNB()
            naive.fit(X_train, y_train)

            y_pred = naive.predict(X_test)

        except Exception as e:
            print("naive_model Error :", e)

        else:
            # print("Naive Bayes Model buit successfully")
            return y_pred
    
    def cm_accuracy(self, y_test, y_pred):
        """performance Metrics"""
        try:
            skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize = (7,7) )
            plt.savefig('CM.jpg')
            img_cm = Image.open("CM.jpg")
            accuracy = accuracy_score(y_test, y_pred)

        except Exception as e:
            print("cm_accuracy error :", e)

        else:
            # print("cm_accuracy poltted successfully")
            return accuracy, img_cm
    
    def word_cloud(self, corpus):
        """Generating Word Cloud"""
        try:
            wordcloud = WordCloud(background_color = 'white', width = 720, height = 500, ).generate(" ".join(corpus))
            plt.imshow(wordcloud, interpolation = "bilinear")
            plt.axis('off')
            plt.savefig('WC.jpg')
            img = Image.open("WC.jpg")

        except Exception as e:
            print("word_cloud error :", e)

        else:
            # print("word cloud plotted")
            return img
        
    def sentimental_analysis_clean(self, text):
        try:
            text = re.sub('http', "", text)
            text = re.sub('co', "", text)
            text = re.sub('amp', "", text)
            text = re.sub('new', "", text)
            text = re.sub('one', "", text)
            text = re.sub('@[A-Za-z0-9]', "", text)
            text = re.sub('#', "", text)
            text = re.sub('RT[\s]+', "", text)
            text = re.sub('http?:\/\/S+', "", text)

            return text
        
        except Exception as e:
            print("sentimental_anaysis_clean error :", e)

    
                 
    


