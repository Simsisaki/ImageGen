from flask import render_template
from flaskexample import app
import psycopg2
from flask import request
import pickle
import re
from collections import Counter
from operator import itemgetter
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from .medium_models import lemmatize, add_phrases, getLDASims, preprocess_lda_post, preprocess_lda_para, doc2vec_paragraph_score
from .medium_utilities import save_obj, load_obj
import gensim
stopwords = load_obj('stopwords')

@app.route('/')
def post_input():
    methods=['GET','POST']
    return render_template("medium_input.html")

@app.route('/medium_output', methods=['GET','POST'])
def post_output():

    if request.method == 'POST':
    
        keyword = request.form.get('keyword')
        doc2vec = gensim.models.doc2vec.Doc2Vec.load('/Users/Simsisaki/Desktop/Insight/Medium/{}_doc2vec'.format(keyword))
        lda_post = gensim.models.LdaModel.load('/Users/Simsisaki/Desktop/Insight/Medium/{}_lda_model_posts'.format(keyword))
        lda_para = gensim.models.LdaModel.load('/Users/Simsisaki/Desktop/Insight/Medium/{}_lda_model_para'.format(keyword))
        img_to_para_dict = load_obj('{}_img_to_para_dict'.format(keyword))
        img_to_kword_dict = load_obj('{}_img_to_kword_dict'.format(keyword))
        id_to_index = load_obj('{}_id_to_index'.format(keyword))
        index_to_id = load_obj('{}_index_to_id'.format(keyword))
        lda_dict_para = load_obj('{}_lda_para_dict'.format(keyword))
        lda_dict_post = load_obj('{}_lda_post_dict'.format(keyword))
        lda_index_post = load_obj('{}_lda_index_post'.format(keyword))    
        lda_index_para = load_obj('{}_lda_index_para'.format(keyword))  
        para_bigram_list = load_obj('{}_para_bigram_list'.format(keyword))

        paragraphs = request.form.get('paragraphs').split("\n")
        paragraphs = [re.sub('\s+$','',paragraph) for paragraph in paragraphs]
        paragraphs = [paragraph for paragraph in paragraphs if paragraph != '']
        all_images = []
        post = ''

        for paragraph in paragraphs:
            post = post + ' ' + paragraph
        post_processed = preprocess_lda_post(post, lemmatizer, stopwords)
        lda_post_sims = getLDASims(lda_index_post, post_processed, lda_dict_post, lda_post)
        
        results = []

        for paragraph in paragraphs[:-1]:
            good_image = False
            if len(paragraph.split(' ')) >= 12:
                doc2vec_sims = doc2vec_paragraph_score(doc2vec,paragraph, lemmatizer, stopwords)
                para_processed = preprocess_lda_para(paragraph, lemmatizer, stopwords, para_bigram_list)
                lda_para_sims = getLDASims(lda_index_para, para_processed, lda_dict_para, lda_para)
                scores = []
                for i in doc2vec_sims:
                    post_index = id_to_index[index_to_id[i[0]]]
                    scores.append([i[0],((.55 * i[1] + .25 * lda_post_sims[post_index]  + .25 * lda_para_sims[i[0]]))])
                
                top_scores = sorted(scores, key=itemgetter(1), reverse=True)[:10]
                best_scores = top_scores[:3]
      
                if best_scores[0][1] >= .635:
                    image_url = img_to_para_dict[best_scores[0][0]]
                    if image_url not in all_images:
                        good_image = True
                        all_images.append(image_url)
                        results.append([paragraph, image_url])
                    elif best_scores[1][1] >= .6:
                        image_url = img_to_para_dict[best_scores[1][0]]
                        if image_url not in all_images:
                            good_image = True
                            all_images.append(image_url)
                            results.append([paragraph, image_url])
                        elif best_scores[2][1] >= .6:
                            image_url = img_to_para_dict[best_scores[2][0]]
                            if image_url not in all_images:
                                good_image = True
                                all_images.append(image_url)
                                results.append([paragraph, image_url])

                if good_image == False:
                    results.append([paragraph,None])
        
            else:    
                results.append([paragraph,None])

        results.append([paragraphs[-1],None])

    return render_template("medium_output.html",results=results)