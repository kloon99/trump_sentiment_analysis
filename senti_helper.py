#Kok Mun Loon 31 Jan 2021
import os
import re
import numpy as np
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from collections import Counter
import string, pickle
from wordcloud import WordCloud
import senti_helper as helper
import matplotlib.pyplot as plt
import plotly.express as px
from colorama import Fore, Back, Style

from transformers import pipeline
nlp = pipeline("sentiment-analysis")


TRANS_MIN_NEG = None
TRANS_MIN_POS = None


from nltk.corpus import stopwords
STOPWORD = stopwords.words('english')

def get_df_for_wc(trump_file_objects):
    trump_wc_df = helper.get_senti_frame('sents', trump_file_objects['trump_sents'])
    trump_wc_df['scores'] = trump_wc_df['sents'].apply(helper.get_sentiments_vals, 
                                                   args=(trump_file_objects['positive_words'],
                                                         trump_file_objects['negative_words']))
    trump_wc_df['negative_count'] = trump_wc_df['scores'].apply(lambda x: x[0])
    trump_wc_df['neutral_count'] = trump_wc_df['scores'].apply(lambda x: x[1])
    trump_wc_df['positive_count'] = trump_wc_df['scores'].apply(lambda x: x[2])
    return trump_wc_df     

def get_plot_for_wc(trump_file_objects):
    trump_wc_df = get_df_for_wc(trump_file_objects)
    sents_neg_count = trump_wc_df.sort_values(by='negative_count', ascending=True)[:50]
    sents_pos_count = trump_wc_df.sort_values(by='positive_count', ascending=False)[:50] 
    return (sents_neg_count, sents_pos_count)            


def get_senti_wcd(df, top_num_words, sentiment='pos', sort_by='sentiment', repeat=True, 
                  _ascending=True, collocation_threshold=1, _repeat=True, 
                  _stopwords=True, _width=600,
                    _height=300):
    _trump_senti_df = None
    if sentiment == 'pos':
        _trump_senti_df =  df.loc[df['sentiment'] > 0] 
    if sentiment == 'neg':
        _trump_senti_df =  df.loc[df['sentiment'] < 0] 
    
    _trump_senti_df =  _trump_senti_df.sort_values(by=sort_by, ascending=_ascending)
    
    #trump_senti_df.loc[trump_senti_df['sentiment'] > 0] 
    #trump_senti_pos_count.sort_values(by=sort_by)
    _sents = _trump_senti_df['sents'].tolist()
    _words_list = list(word_tokenize(_sent ) for _sent in _sents)
    pos_combined_wds_list = []
    for _words in _words_list: 
        for _wd in _words:
            pos_combined_wds_list.append(_wd)
    
    #print(pos_combined_wds_list)
    xc  = Counter(pos_combined_wds_list).most_common(top_num_words)
    xcd = {_c[0]:_c[1] for _c in xc if _c[0] not in string.punctuation and  _c[0].lower() not in STOPWORD}

    wc = WordCloud(background_color="white", collocation_threshold=collocation_threshold, repeat=_repeat, stopwords=STOPWORD,
                   width=_width, height=_height)
    # generate word cloud
    wc.generate_from_frequencies(xcd)

    #import matplotlib.pyplot as plt
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    
def get_sentiments_raw(sent, pos, neg):
    score = 0
    for word in word_tokenize(sent):
        if word in pos:
            score += 1     
        elif word in neg:
            score -= 1
        else:
            #print(word)
            pass
    return score  
    

def get_sentiments_vals(sent, pos_w, neg_w):
    neu = 0
    pos = 0
    neg = 0
    for word in word_tokenize(sent):
        if word in pos_w:
            pos += 1     
        elif word in neg_w:
            neg -= 1
        else:
            neu += 1    
    return (neg, neu, pos)     
    
def words_fm_txt_no_punct(_text):

    _words = word_tokenize(_text)
    _words = [wd for wd in _words if wd.lower() not in string.punctuation ]
    #_words = [wd for wd in trump_words if wd not in r'.,?' ]
    
    return _words
    

def get_senti_frame(_col_nm, _data):
    return pd.DataFrame({   
    _col_nm : _data
}   )

def ploty_fig_wc(df, senti):
    fig = None
    _color = None
    _yaxis = None
    if senti =='neg':
        _color = "negative_count"
        _yaxis = "negative_count"
    if senti =='pos':
        _color = "positive_count"
        _yaxis = "positive_count"
    
    fig = px.bar(df, x=range(50), y=_yaxis,
            hover_data=['sents'], 
            labels={'sents':''}, 
            height=400, color=_color, 
            title="negative word counts")
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="arial narrow"
    ))
    return fig

def display_process_sents(df, sort_by, col, ascending=True, num_sents=10):
    sents_display = df.sort_values(by=sort_by, ascending=ascending)[:num_sents][col]
    trump_speech_sents = [word_tokenize(sent) for sent in sents_display.to_list()]
    return trump_speech_sents


def get_disp_sent(speech_sents):
    wds_list = []
    for sent in speech_sents:
        all_wds_neg = ''
        for word in sent:
            #Rid punctuations
            if word not in string.punctuation:
                all_wds_neg += ' ' + word 
            else:
                all_wds_neg += word
        all_wds_neg = all_wds_neg.strip()
        wds_list.append(all_wds_neg)
    return wds_list

#This is for terminal colors.
def get_color_sents(wds_list, pos_wds, neg_wds):
    #from colorama import Fore, Back, Style
    color_sents_list = list()
    for sent in wds_list:
        color_speech = ''
        for word in word_tokenize(sent): 
            _txt = ' ' + word if word not in string.punctuation else word
            #_txt =  word
            if word.lower().strip() in pos_wds:
                color_speech += Fore.BLUE + _txt 
            elif word.lower().strip() in neg_wds:
                color_speech += Fore.RED + _txt 
            else: 
                color_speech += Fore.BLACK + _txt 
        color_sents_list.append(color_speech)
    return color_sents_list


def get_color_sents_htm(wds_list,pos_wds, neg_wds):
    #from colorama import Fore, Back, Style
    #color_sents_list = list()
    color_speech_all = ''
    for sent in wds_list:
        color_speech = ''
        for word in word_tokenize(sent): 
            _txt = ' ' + word if word not in string.punctuation else word
            #_txt =  word
            #import pdb; pdb.set_trace()
            if word.lower().strip() in pos_wds:
                #color_speech += Fore.BLUE + _txt 
                color_speech += r'<span class="{}">{}</span>'.format('blue1',_txt)
            elif word.lower().strip() in neg_wds:
                #color_speech += Fore.RED + _txt 
                color_speech += r'<span class="{}">{}</span>'.format('red1',_txt)
            else: 
                #color_speech += Fore.BLACK + _txt 
                #color_speech += r' <span class="{}">{}</span> '.format('black1',word)
                color_speech +=   _txt
        color_speech_all = color_speech_all + r'<p>' + color_speech + r'</p>' 
        #color_sents_list.append(color_speech)
    return color_speech_all


#credit: https://www.saltycrane.com/blog/2007/09/python-word-wrap-function/
def word_wrap(string, width=80, ind1=0, ind2=0, prefix=''):
    """ word wrapping function.
        string: the string to wrap
        width: the column number to wrap at
        prefix: prefix each line with this string (goes before any indentation)
        ind1: number of characters to indent the first line
        ind2: number of characters to indent the rest of the lines
    """
    string = prefix + ind1 * " " + string
    newstring = ""
    while len(string) > width:
        # find position of nearest whitespace char to the left of "width"
        marker = width - 1
        while not string[marker].isspace():
            marker = marker - 1

        # remove line from original string and add it to the new string
        newline = string[0:marker] + "\n"
        newstring = newstring + newline
        string = prefix + ind2 * " " + string[marker + 1:]

    return newstring + string


def get_transform_pd(sents):
    trump_transform_df = get_senti_frame('sents',sents)
    trump_transform_df.drop_duplicates(inplace=True, ignore_index=True)
    trump_transform_sents = trump_transform_df['sents'].tolist()
    trump_transform_sents_results = nlp(trump_transform_sents)
    trump_transform_df['label'] = [_itm['label'] for _itm in trump_transform_sents_results]
    trump_transform_df['score'] = [_itm['score'] for _itm in trump_transform_sents_results]
    return trump_transform_df

def scale_index_neg(score, _min):
    return -((((score - _min)*10000)**2))-1
    

def scale_index_pos(score, _min):
    return (((score - _min)*100000)**2)+10
#TRANS_MIN_POS = np.min(trump_trans_pos_df['score'])


def get_trump_trans_neg_df(trump_transform_df):

    trump_trans_neg_df = trump_transform_df.loc[trump_transform_df['label'] == 'NEGATIVE'].sort_values(by='score', ascending=False )[:50]
    TRANS_MIN_NEG = np.min(trump_trans_neg_df['score'])
    trump_trans_neg_df['score_scaled'] = trump_trans_neg_df['score'].apply(scale_index_neg, args=(TRANS_MIN_NEG,))

    return trump_trans_neg_df

def trump_trans_neg_df_plot(trump_trans_neg_df):
    fig = px.bar(trump_trans_neg_df, x=range(trump_trans_neg_df.shape[0]), y="score_scaled",
             hover_data=['sents'], 
             labels={'sents':'sents'}, height=400, color="score", title="sentiment")
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="arial narrow"
        )
    )
    fig.show()
    return fig


def get_trump_trans_pos_df(trump_transform_df):
    trump_trans_pos_df = trump_transform_df.loc[trump_transform_df['label'] == 'POSITIVE'].sort_values(
        by='score', ascending=False )[:50]
    TRANS_MIN_POS = np.min(trump_trans_pos_df['score'])
    trump_trans_pos_df['score_scaled'] = trump_trans_pos_df['score'].apply(scale_index_pos, args=(TRANS_MIN_POS,))

    return trump_trans_pos_df



def trump_trans_pos_df_plot(trump_trans_pos_df):
    fig = px.bar(trump_trans_pos_df, x=range(trump_trans_pos_df.shape[0]), y="score_scaled",
             hover_data=['sents'], 
             labels={'sents':'sents'}, height=400, color="score", title="sentiment")
    
    fig.update_layout(
        hoverlabel=dict(
        bgcolor="white",
        font_size=11,
        font_family="arial narrow"
        )
    )
    fig.show()

    return fig