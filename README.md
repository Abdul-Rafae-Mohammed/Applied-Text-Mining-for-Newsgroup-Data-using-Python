# Applied-Text-Mining-for-Newsgroup-Data-using-Python
There are 2 Parts in this Project – Part-1: Document Similarity, Part-2: Topic Modelling.

Part-1: I used the Paraphrase Dataset to find out if one text is a paraphrase of another. Here I used the NLTK library’s 
WORDNET API to find the document similarity.
Part-2: I used the Newsgroup Dataset to perform Topic Modelling by applying the Latent 
Dirichlet Allocation (LDA) model to topic model the dataset.



## Project - Document Similarity & Topic Modelling

### Part 1 - Document Similarity

#There are following functions :
#`convert_tag:`** converts the tag given by `nltk.pos_tag` to a tag used by `wordnet.synsets`. 
#`document_path_similarity:`** computes the symmetrical path similarity between two documents by finding the synsets in each document #using `doc_to_synsets`, then computing similarities using `similarity_score`.
#`doc_to_synsets:`** returns a list of synsets in document. This function should first tokenize and part of speech tag the document using `nltk.word_tokenize` and `nltk.pos_tag`. Then it will find each tokens corresponding synset using `wn.synsets(token, wordnet_tag)`. The first synset matchis used. If there is no match, that token is skipped.
#`similarity_score:`** returns the normalized similarity score of a list of synsets (s1) onto a second list of synsets (s2). For each synset in s1, find the synset in s2 with the largest similarity value. Sum all of the largest similarity values together and normalize this value by dividing it by the number of largest similarity values found.  Missing values are ignored.


### Part 2 - Topic Modelling
# 
#For the second part, I used Gensim's LDA (Latent Dirichlet Allocation) model to model topics in `newsgroup_data`.

