#Remove stopwords and stemmers
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(Words)):
    review = re.sub('[^a-zA-Z]', ' ', Words['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)




# basic statistical features
df['word_count'] = df["text"].apply(lambda x: len(str(x).split(" ")))
df['char_count'] = df["text"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
df['avg_word_length'] = df['char_count'] / df['word_count']

df['sentence_count'] = df["text"].apply(lambda x: len(str(x).split(".")))
df['avg_sentence_length'] = df['word_count'] / df['sentence_count']

Words['STD_Ratings']=Words['Ratings'].apply(lambda x: (x-Words['Ratings'].mean())/Words['Ratings'].std())


#cosine similarity based on tf-idf matrix 
# tf-idf matrix input as a dataframe
from sklearn.metrics.pairwise import cosine_similarity  

# X = np.array(tf_idf)
cos=cosine_similarity(np.array(tf_idf) ,np.array(tf_idf))
I = np.identity(4)
Cosine_Matrix_tfidf=(cosine_similarity(np.array(tf_idf) ,np.array(tf_idf))-np.identity(tf_idf.shape[0]))
df['Cosine_AVG']= Cosine_Matrix_tfidf.mean(axis=0)
df['Cosine_STD']= Cosine_Matrix_tfidf.std(axis=0)


#Jacquard similarity based on list of words
#input is a list of text words should be stemmed

def jaccard(a, b):
    a = set(a.split())
    b = set(b.split())
    return float(len(a & b)) / len(a | b)

JaccardMatrix = np.zeros((len(Words),len(Words)))

for  i in range (0,len(Words)):
    for  j in range (0,len(Words)):
        if i!=j :
            JaccardMatrix[i,j]=jaccard(Words[i],Words[j])
            
df['Jaccard_Similarity_AVG']= JaccardMatrix.mean(axis=0)
df['Jaccard_Similarity_STD']= JaccardMatrix.std(axis=0)            
 



