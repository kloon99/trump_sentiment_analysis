# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python (transformer1)
#     language: python
#     name: transformer
# ---

# %%
#Kok Mun Loon 14 Feb 2021
import os
import re
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from nltk import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
import senti_helper as helper

# %%
from wordcloud import WordCloud
from textblob import TextBlob

# %%
# %load_ext autoreload
# %autoreload 2


# %%
pd.set_option('max_colwidth', 1200)

# %%
from nltk.corpus import stopwords
STOPWORD = stopwords.words('english')

# %%
import string
string.punctuation
PUNCT = r'!"#$%&()*+-/<=>?@[]\^_`{|}~‘’' 

# %% code_folding=[0] language="html"
# <style>
# .ex1 {
#   font-family: Verdana, sans-serif;
#   margin: auto;
#   border: 2px solid #73AD21;
#   padding: 5px 5px 5px 5px;
# </style>

# %% [markdown]
# <div class="ex1">Since I have already downloaded the necessary datasets and we can simply load this into the notebook.
# </div>

# %%
trump_file_objects = pickle.load(open('trump_file_objects.pkl', mode='rb'))

# %% [markdown]
# <div class="ex1">For our first task, for every sentence, we assign a score of -1 for each negative word and a score of +1 for each positive word. The total sentiment score for a sentennce is simply the total of the postive word and negative words scores. We can then feed the resutling list into a Python Counter object and n plot a histogram of the counts of the total sentiment score. </div>

# %%
total_sentiment_score = [helper.get_sentiments_raw(sent,trump_file_objects['positive_words'],
                                                   trump_file_objects['negative_words']) for sent in trump_file_objects['trump_sents']]

# %%
#Usisg the Counter object to process the counts of the different score
sentiment_count = Counter(total_sentiment_score)
for _count in sorted(sentiment_count.keys()):
    print("Score of: {}; Total count is {}.".format(_count,sentiment_count[_count]))

# %%
#Create datafram for plotting.
total_sentiment_score_df = pd.DataFrame({
    'score' : total_sentiment_score
})

# %%
#Plot bar chart using plotly express.
import plotly.express as px
fig = px.histogram(total_sentiment_score_df, x='score', color='score', template='plotly_white', title='Sentence Scores')
fig.show()

# %% [markdown]
# Next, we get a plot of the 50 most negative sentences and a correpodng plot of the most postive sentences. Using plotly express we can see a bar chart of these sentences as well as the actual text comprised within with a mouse hover.

# %%
sents_neg_count, sents_pos_count = helper.get_plot_for_wc(trump_file_objects)

# %%
fig_wc_neg = helper.ploty_fig_wc(sents_neg_count, 'neg')
fig_wc_neg.show()

# %% [markdown]
# The theme of these negative-sentiment sentences centers around the alleged voter fraud and how the election was alleged stolen from Mr. Trump. Blamed as also placed on news media and that voting machines that were used for the US 2020 elections. This is contrasted with the positive-sentiment sentences in which Mr. Trump expresses gratitude to his supporters.

# %%
fig_wc_pos = helper.ploty_fig_wc(sents_pos_count, 'pos')
fig_wc_pos.show()

# %% [markdown]
# Often, it is useful to see the text with the key words highlighted. We can do this by creating a basic style sheet to display the text as a HTML snippet. Here in this sample of 5 setences, red means a negative word, blue means a positive word and neutral words are displayed in the default black font. 

# %%
NEG_WORDS = list(wd.lower() for wd in trump_file_objects['negative_words'])
POS_WORDS = list(wd.lower() for wd in trump_file_objects['positive_words'])

# %%
#Process setences as HTML code with color key words highlighting.
trump_wc_df = helper.get_df_for_wc(trump_file_objects)
display_neg_sents = helper.get_disp_sent(helper.display_process_sents(trump_wc_df, 'negative_count', 'sents', 
                                                                      ascending=True, num_sents=10))
color_sents_list = helper.get_color_sents_htm(display_neg_sents,POS_WORDS, NEG_WORDS)

# %% code_folding=[1, 2]
from IPython.core.display import HTML
display(HTML(
    r'''<style>
        .blue1 {  
          color: blue;
          margin: 0px;
          padding: 0px;
        }
        .red1 {  
          color: red;
          margin: 0px;
          padding: 0px;
        }
        .black1 {  
          color: black;
          margin: 0px;
          padding: 0px;
        }
    </style>'''
))
HTML(color_sents_list)

# %%
#Create sentiment dataframe
trump_senti_df = helper.get_senti_frame('sents', trump_file_objects['trump_sents'])
trump_senti_df.head()

# %% [markdown]
# <div class="ex1">The sentiment function of textblob returns two properties, polarity, and subjectivity.
# Polarity is float between 1 and -1 where 1 is the most positive and -1 means is the most negative. Subjectivity is in turn a float between 0 and 1, where 1 is the highest score indicating subjectivity. 
# </div>

# %%
print('I am a happy camper:', TextBlob('I am a happy camper').sentiment)
print('I am a sad sack: ', TextBlob('I am a sad sack').sentiment)
print('The captital of Italy is Rome: ',TextBlob('The captital of Italy is Rome').sentiment)

# %%
trump_senti_df['sentiment'] = trump_senti_df['sents'].apply(lambda x: TextBlob(x).polarity)

# %%
trump_senti_df['subjectivity'] = trump_senti_df['sents'].apply(lambda x: TextBlob(x).subjectivity)

# %%
trump_senti_df.head()

# %%
STOPWORD = STOPWORD + ['going', 'away', 'get', 'dont', 'one', 'way', 'want', 'go', 'said', 'much', 'say']

# %%
helper.get_senti_wcd(trump_senti_df, top_num_words=50, sentiment='neg', _ascending=False, _stopwords=STOPWORD)

# %%
helper.get_senti_wcd(trump_senti_df, top_num_words=50, sentiment='pos',_ascending=True, _stopwords=STOPWORD)

# %%
from transformers import pipeline
nlp = pipeline("sentiment-analysis")
result = nlp(["I hate you", 'I love you'])
print(result)

# %%
trump_transform_df = helper.get_transform_pd(trump_file_objects['trump_sents'])

# %%
trump_transform_df.head()

# %%
trump_trans_neg_df = helper.get_trump_trans_neg_df(trump_transform_df)
trump_trans_neg_df.head(10)

# %%
trump_trans_neg_fig = helper.trump_trans_neg_df_plot(trump_trans_neg_df)

# %%
trump_trans_pos_df = helper.get_trump_trans_pos_df(trump_transform_df)
trump_trans_pos_df.head(10)

# %%
trump_trans_pos_fig = helper.trump_trans_pos_df_plot(trump_trans_pos_df)
