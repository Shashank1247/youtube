from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import pandas as pd
import demoji
from langdetect import detect
import re   # regular expression
import numpy as np
import os
import os
import scipy
import warnings
from textblob import TextBlob
from langdetect import detect
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from flask import Flask,render_template

# key="AIzaSyAIGeU6DMwurpAb9Ndcwyr9TB2rVzzPnbk"

# def build_service():
#     YOUTUBE_API_SERVICE_NAME = "youtube"
#     YOUTUBE_API_VERSION = "v3"
#     return build(YOUTUBE_API_SERVICE_NAME,
#                  YOUTUBE_API_VERSION,
#                  developerKey=key)

# service = build_service()

# query = "Tech Q&A Session Your Questions Answered March 2021"

# query_results = service.search().list(part='snippet', q=query,
#                                       order='relevance',
#                                       type='video',
#                                       relevanceLanguage='en',
#                                       safeSearch='moderate').execute()


# video_id = []
# channel = []
# video_title = []
# video_desc = []


# for item in query_results['items']:
#     video_id.append(item['id']['videoId'])
#     channel.append(item['snippet']['channelTitle'])
#     video_title.append(item['snippet']['title'])
#     video_desc.append(item['snippet']['description'])

# video_id = video_id[0]
# channel = channel[0]
# video_title = video_title[0]
# video_desc = video_desc[0]


# video_id_pop = []
# channel_pop = []
# video_title_pop = []
# video_desc_pop = []
# comments_pop = []
# comment_id_pop = []
# reply_count_pop = []
# like_count_pop = []


# comments_temp = []
# comment_id_temp = []
# reply_count_temp = []
# like_count_temp = []

# nextPage_token = None

# while 1:
#   response = service.commentThreads().list(
#                     part = 'snippet',
#                     videoId = video_id,
#                     maxResults = 100, 
#                     order = 'relevance', 
#                     textFormat = 'plainText',
#                     pageToken = nextPage_token
#                     ).execute()


#   nextPage_token = response.get('nextPageToken')
#   for item in response['items']:
#       comments_temp.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
#       comment_id_temp.append(item['snippet']['topLevelComment']['id'])
#       reply_count_temp.append(item['snippet']['totalReplyCount'])
#       like_count_temp.append(item['snippet']['topLevelComment']['snippet']['likeCount'])
#       comments_pop.extend(comments_temp)
#       comment_id_pop.extend(comment_id_temp)
#       reply_count_pop.extend(reply_count_temp)
#       like_count_pop.extend(like_count_temp)
        
#       video_id_pop.extend([video_id]*len(comments_temp))
#       channel_pop.extend([channel]*len(comments_temp))
#       video_title_pop.extend([video_title]*len(comments_temp))
#       video_desc_pop.extend([video_desc]*len(comments_temp))

#   if nextPage_token is  None:
#     break

# # print(comments_pop)

# output_dict = {
#         'Channel': channel_pop,
#         'Video Title': video_title_pop,
#         'Video Description': video_desc_pop,
#         'Video ID': video_id_pop,
#         'Comment': comments_pop,
#         'Comment ID': comment_id_pop,
#         'Replies': reply_count_pop,
#         'Likes': like_count_pop,
#         }

# output_df = pd.DataFrame(output_dict, columns = output_dict.keys())

# duplicates = output_df[output_df.duplicated("Comment ID")]


# unique_df = output_df.drop_duplicates(subset=['Comment'])

# unique_df.to_csv("extraced_comments.csv",index = False)

# comments = pd.read_csv('extraced_comments.csv')

# demoji.download_codes()

# comments['clean_comments'] = comments['Comment'].apply(lambda x: demoji.replace(x,""))

# comments['language'] = 0

# count = 0
# for i in range(0,len(comments)):


#   temp = comments['clean_comments'].iloc[i]
#   count += 1
#   try:
#     comments['language'].iloc[i] = detect(temp)
#   except:
#     comments['language'].iloc[i] = "error"

# comments[comments['language']=='en']['language'].value_counts()

# english_comm = comments[comments['language'] == 'en']

# english_comm.to_csv("english_comments.csv",index = False)

# en_comments = pd.read_csv('english_comments.csv',encoding='utf8',error_bad_lines=False)

# en_comments.shape

# regex = r"[^0-9A-Za-z'\t]"

# copy = en_comments.copy()

# copy['reg'] = copy['clean_comments'].apply(lambda x:re.findall(regex,x))
# copy['regular_comments'] = copy['clean_comments'].apply(lambda x:re.sub(regex,"  ",x))

# dataset = copy[['Video ID','Comment ID','regular_comments']].copy()


# dataset = dataset.rename(columns = {"regular_comments":"comments"})

# dataset.to_csv("Dataset.csv",index = False)

data = pd.read_csv("Dataset.csv")

data['polarity'] = data['comments'].apply(lambda x: TextBlob(x).sentiment.polarity)

data = data.sample(frac=1).reset_index(drop=True)

data['pol_cat']  = 0

data['pol_cat'][data.polarity > 0] = 1
data['pol_cat'][data.polarity <= 0] = -1
data['pol_cat'].value_counts()
data_pos = data[data['pol_cat'] == 1]
data_pos = data_pos.reset_index(drop = True)
data_neg = data[data['pol_cat'] == -1]
data_neg = data_neg.reset_index(drop = True)

data_neg['comments'][40]


data['comments'] = data['comments'].str.lower()

nltk.download("stopwords")


nltk.download("punkt")

stop_words = set(stopwords.words('english'))

data['comments'] = data['comments'].str.strip()

train = data.copy()

train['comments'] = train['comments'].str.strip()


def remove_stopwords(line):
    word_tokens = word_tokenize(line)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence)

data['stop_comments'] = data['comments'].apply(lambda x : remove_stopwords(x))


X_train,X_test,y_train,y_test = train_test_split(data['stop_comments'],data['pol_cat'],test_size = 0.2,random_state = 324)

data['pol_cat'].value_counts()

vect = CountVectorizer()
tf_train = vect.fit_transform(X_train)
tf_test = vect.transform(X_test)

vocab = vect.vocabulary_

lr = LogisticRegression()
lr.fit(tf_train,y_train)

lr.score(tf_test,y_test)


expected = y_test
predicted = lr.predict(tf_test)


cf = metrics.confusion_matrix(expected,predicted,labels = [1,-1])
print(cf)
fig, ax = plot_confusion_matrix(conf_mat = cf)
print(plt.show())
print(metrics.classification_report(expected, predicted))
print(f1_score(expected, predicted, average='macro'))