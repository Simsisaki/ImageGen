import gensim
from operator import itemgetter
from nltk.stem.wordnet import WordNetLemmatizer


def lemmatize(doc,lemmatizer):
	# Lemmatize words in doc
	lemma_doc = [lemmatizer.lemmatize(token) for token in doc]
	return lemma_doc


def add_phrases(doc,bigram_list):
	# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
	for token in bigram_list[doc]:
		if '_' in token:
			# Token is a bigram, add to document.
			doc.append(token)
	return doc


def getLDASims(index, doc, dictionary,model):
	doc_bow = dictionary.doc2bow(doc)
	doc_lda = model[doc_bow]
	sims = index[doc_lda]
	sim_vectors =  sorted(enumerate(sims), key=lambda item: -item[1])
	return {vec[0]:vec[1] for vec in sim_vectors}


def preprocess_lda_post(post, lemmatizer, stopwords):
	processed = gensim.utils.simple_preprocess(post)
	processed_stop = [word for word in processed if word not in stopwords]
	lemma_doc = lemmatize(processed_stop,lemmatizer)
	return lemma_doc


def preprocess_lda_para(paragraph,lemmatizer, stopwords, bigram_list):
	processed = gensim.utils.simple_preprocess(paragraph)
	processed_stop = [word for word in processed if word not in stopwords]
	lemma_doc = lemmatize(processed_stop,lemmatizer)
	return add_phrases(lemma_doc,bigram_list) 
	  
			
def doc2vec_paragraph_score(model, paragraph, lemmatizer, stopwords):
	processed = gensim.utils.simple_preprocess(paragraph)
	processed_stop = [word for word in processed if word not in stopwords]
	lemma_doc = lemmatize(processed_stop,lemmatizer)
	inferred_vector = model.infer_vector(lemma_doc)
	top_scores = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))[:500]
	return top_scores


