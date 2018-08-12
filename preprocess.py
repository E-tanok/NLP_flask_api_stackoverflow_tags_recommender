from context import dir_path, datasources_path, learning_models_path

import re
import pandas as pd
import pickle
import numpy as np
import nltk
nltk.data.path.append(dir_path+'/nltk_data/')
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('universal_tagset')
#nltk.download('wordnet')
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords # Import the stop word list
import time



from functions import smart_lemmatize
from tailored_functions import find_similar_neighbors, generate_tags

#Words string management :
L_frequent_tags = pickle.load(open(datasources_path+"L_frequent_tags.p", "rb" ))
L_words_of_tags = pickle.load(open(datasources_path+"L_words_of_tags.p", "rb" ))
corpus_stopwords = pickle.load(open(datasources_path+"corpus_stopwords.p", "rb" ))
dict_tags_replacement = pickle.load(open(datasources_path+"dict_tags_replacement.p", "rb" ))
reverse_dict_tags_replacement = {v: k for k, v in dict_tags_replacement.items()}

english_stopwords  = stopwords.words("english")
L_replace_text = ['[\?!\.,;:=@%]','<\w{1,10}>','</\w{1,10}>','\\\\\w{1,10}',
'\'s',"['`]",'"','[\(\)\{\}]','[<>]','href','https//','http//','/','-','_',
'enwikipediaorgwiki','wikipediaorgwiki','enwikipediaorg','wikipediaorg',
'enwikipedia','wikipedia','enwiki','wiki']


#Models & recommendation management :
lda_model = pickle.load(open(learning_models_path+"lda_model.p", "rb" ))
df_train_words_space_schema = pickle.load(open(datasources_path+"df_train_words_space_schema.p", "rb" ))
dict_topics_tags = pickle.load(open(datasources_path+"dict_topics_tags.p", "rb" ))
dict_topics_words = pickle.load(open(datasources_path+"dict_topics_words.p", "rb" ))
df_lda_words_similarity = pickle.load(open(datasources_path+"df_lda_words_similarity.p", "rb" ))
df_lda_tags_similarity = pickle.load(open(datasources_path+"df_lda_tags_similarity.p", "rb" ))
df_word2vec_words_similarity = pickle.load(open(datasources_path+"df_word2vec_words_similarity.p", "rb" ))
df_word2vec_tags_similarity = pickle.load(open(datasources_path+"df_word2vec_tags_similarity.p", "rb" ))
df_word2vec_pure_tags_similarity = pickle.load(open(datasources_path+"df_word2vec_pure_tags_similarity.p", "rb" ))

def normalize_and_clean(my_string):
    start_time = time.time()
    print("Normalization and Cleaning ...")
    my_string = str.lower(my_string)
    for expr in L_replace_text:
        my_string = str.replace(my_string,expr,'')
    my_string = re.sub('[-\.]','',my_string)
    my_string = re.sub('[cC] {0,1}#','csharp',my_string)
    my_string = str.replace(my_string,r'\r',' ')
    my_string = str.replace(my_string,r'\n',' ')

    print("Tokenization ...")
    my_tokenized_string = word_tokenize(my_string)
    print(my_tokenized_string)

    print("English stopwords processing ...")
    for stopword in english_stopwords :
        my_tokenized_string = list(filter(lambda a: a != stopword, my_tokenized_string))
    print(my_tokenized_string)

    print("Lemmatization ...")
    my_tokenized_string = smart_lemmatize(my_tokenized_string,L_frequent_tags+L_words_of_tags)
    print(my_tokenized_string)

    print("Corpus stopwords processing ...")
    for stopword in corpus_stopwords :
        my_tokenized_string = list(filter(lambda a: a != stopword, my_tokenized_string))
    print(my_tokenized_string)
    my_clean_string = ' '.join(my_tokenized_string)
    print(my_clean_string)
    my_string_to_vectorize = [my_clean_string]

    print("Vectorization ...")
    count_vectorizer = CountVectorizer(max_df=1.0, min_df=0, max_features=1000)
    count = count_vectorizer.fit_transform(my_string_to_vectorize)
    vectorizer_feature_names = count_vectorizer.get_feature_names()
    count_array = count.toarray()
    df_all_words = pd.DataFrame(count_array, columns=vectorizer_feature_names)

    print("Reshaping the dimension on the train words space perimeter ...")
    #The original shape is :
    words_dim = df_all_words.shape[1]
    print("original test space words dimension is %s"%words_dim)
    #We have to filter the data onto the train space perimeter
    df_all_words = df_all_words[[col for col in df_all_words.columns if col in list(df_train_words_space_schema.columns)]]
    words_dim = df_all_words.shape[1]
    print("after filtering on the train columns perimeter, test space words dimension is %s"%words_dim)

    print("Building the final corpus matrix ...")
    df_final_words = df_train_words_space_schema.append(df_all_words)
    df_final_words.fillna(0, inplace=True)
    matrix = df_final_words.values

    print("Applying the LDA model to the corpus ...")
    corpus_topic_distribution = lda_model.transform(matrix)
    print(corpus_topic_distribution)
    message_topic_distribution = corpus_topic_distribution[0]
    message_main_topic = np.argmax(message_topic_distribution)

    print("Generating the LDA eligible Tags and words ...")
    lda_tags = dict_topics_tags[message_main_topic]
    lda_words = dict_topics_words[message_main_topic]

    print("\n\n")
    print("Generating the tags ...")
    tags_recommendation = generate_tags(my_clean_string,
    df_lda_words_similarity,
    df_lda_tags_similarity,
    df_word2vec_words_similarity,
    df_word2vec_tags_similarity,
    df_word2vec_pure_tags_similarity,
    L_frequent_tags,
    5,
    lda_tags,
    0.8)
    tags_recommendation = set(reverse_dict_tags_replacement[tag] for tag in tags_recommendation)

    print("\n\n")
    print("Recommended tags are :")
    print(tags_recommendation)
    print("\n\n")
    print("Execution time : %s seconds"%(time.time() - start_time))
    print("\n\n")
    print("program over ...")

    return(tags_recommendation)
