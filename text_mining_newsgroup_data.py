
# coding: utf-8


import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd


def convert_tag(tag):
    """Converts the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    

    text13 = nltk.word_tokenize(doc)
    #WNlemma = nltk.WordNetLemmatizer()
    #text13 = [WNlemma.lemmatize(t) for t in text13]
    #text13 = nltk.word_tokenize(doc)
    tags = nltk.pos_tag(text13)
    #print(text13, tags[0][0],tags[0][1])
    synset_l = []
    for i in tags:
        #print(i[0], convert_tag(i[1]))
        m = wn.synsets(i[0], convert_tag(i[1]))
        #print(m)
        if m:
            #print(m)
            synset_l.append(m[0])
    return synset_l


def similarity_score(s1, s2):
    """
    Calculates the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    
    
    
    synset_arr = []
    largest_synset =[]
    for i in s1:
        for j in s2:
            #if i!=j:
            synset_arr.append(i.path_similarity(j))
            #print(i,j)
        #print("syn_arr",synset_arr)
        synset_arr = sorted(list(filter(None.__ne__, synset_arr)))
        if synset_arr:
            largest_synset.append(np.float(synset_arr[-1]))
        synset_arr=[]
        #largest_synset.append(sorted(synset_arr)[0])
        #print(largest_synset)
    return np.mean(largest_synset)


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2
#print(doc_to_synsets('I like cats'))
#synsets1 = doc_to_synsets('I like cats')
#synsets2 = doc_to_synsets('I like dogs')
#print("sim_score",similarity_score(synsets1, synsets2))
#document_path_similarity('I like cats', 'I like dogs')


# ### test_document_path_similarity
# 
# Checks if doc_to_synsets and similarity_score are correct.


def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets     and similarity_score is correct!'
    return document_path_similarity(doc1, doc2)
test_document_path_similarity()



# `paraphrases` is a DataFrame which contains the following columns: `Quality`, `D1`, and `D2`.
# 
# `Quality` is an indicator variable which indicates if the two documents `D1` and `D2` are paraphrases of one another (1 for paraphrase, 0 for not paraphrase).


paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head()

# ### most_similar_docs
# 
# Using `document_path_similarity`, find the pair of documents in paraphrases which has the maximum similarity score.
# 
# *This function should return a tuple `(D1, D2, similarity_score)`*

D1_list = paraphrases['D1'].tolist()
D2_list = paraphrases['D2'].tolist()
#print(len(D1_list))
sim_list = []
for i in D1_list:
    for j in D2_list:
        sim_list.append((i,j,document_path_similarity(i,j)))
def most_similar_docs():
    
  
    return sorted(sim_list, key=lambda tup: tup[2])[-1]
most_similar_docs()


# ### label_accuracy
# 
# Provides labels for the twenty pairs of documents by computing the similarity for each pair using `document_path_similarity`. Let the classifier rule be that if the score is greater than 0.75, label is paraphrase (1), else label is not paraphrase (0). Report accuracy of the classifier using scikit-learn's accuracy_score.


# In[61]:

def label_accuracy():
    from sklearn.metrics import accuracy_score

    D1_list = paraphrases['D1'].tolist()
    D2_list = paraphrases['D2'].tolist()
    #print(len(D1_list))
    sim_list = []
    for i,j in zip(D1_list, D2_list):
            sim_list.append((i,j,document_path_similarity(i,j)))
    para_list_T = paraphrases['Quality'].tolist()
    para_list_P = []
    for i in sim_list:
        if i[2] > 0.75:
            #print(i[2])
            para_list_P.append(1)
        else:
            #print(i[2])
            para_list_P.append(0)
    #print(len(para_list_T),len(para_list_P))
    return accuracy_score(para_list_T, para_list_P)
label_accuracy()


#Part-2

import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Load the list of documents
with open('newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Using CountVectorizor to find three letter tokens, remove stop_words, 
# removing tokens that don't appear in at least 20 documents,
# removing tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())


# In[4]:

# Use the gensim.models.ldamodel.LdaModel constructor to estimate 
# LDA model parameters on the corpus, and save to the variable `ldamodel`

# Your code here:
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10,id2word = id_map,passes = 25,random_state=34)


# ### lda_topics
# 
# Using `ldamodel`, find a list of the 10 topics and the most significant 10 words in each topic.


# In[22]:

def lda_topics():
    
    
    topic_list = ldamodel.print_topics(num_topics=10, num_words=10)
    return topic_list
lda_topics()


# ### topic_distribution
# 
# For the new document `new_doc`, finding the topic distribution. using vect.transform on the the new doc, and Sparse2Corpus to convert the sparse matrix to gensim corpus.


new_doc = ["\n\nIt's my understanding that the freezing will start to occur because of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge Krumins\n-- "]


# In[27]:

def topic_distribution():
    
    # Your Code Here
    new_X = vect.transform(new_doc)
    new_corpus = gensim.matutils.Sparse2Corpus(new_X, documents_columns=False)
    for i in ldamodel.get_document_topics(new_corpus):
        dist_list = i
    #print(ldamodel.print_topics(new_corpus))
    return dist_list
topic_distribution()



