# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:24:42 2020

@author: elias
"""



from AugmentText import*



import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df= pd.read_csv('emails.csv')
#Visualisation
ham = df[df['spam']==0]
spam = df[df['spam']==1]
alpha=0.1
num_aug=3
df2= pd.read_csv('spam.csv')



# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(df2)):
    review = re.sub('[^a-zA-Z]', ' ', df2['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


for i in range(0, len(df2)):
    sentence = corpus[i]
    aug_sentences = eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)


writer = open('Test1.csv', 'w')
lines = open('spam.csv', 'r').readlines()
for i, line in enumerate(lines):
    parts = line[:-1].split('\t')
    sentence = parts[0]
    print(sentence+'\n')
    #aug_sentences = eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
    #for aug_sentence in aug_sentences:
        #writer.write( aug_sentence + '\n')
writer.close() 
# gen_eda ('spam.csv','Test1.csv',0.1,10)
#lines = df2.readlines()

#generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha, num_aug=9):

    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').readlines()

    for i, line in enumerate(lines):
        
        parts = line[:-1] #.split('\t')
        #label = parts[0]
        sentence = parts[1]
        aug_sentences = eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            writer.write(label + "\t" + aug_sentence + '\n')

    writer.close()
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(num_aug))