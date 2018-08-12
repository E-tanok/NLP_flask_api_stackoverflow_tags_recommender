def smart_lemmatize(myList,myStopwordsList):
    """
    This function goal is to lemmatize the words contained in a list ('myList' arg)

    RESULT : a list of lemmatized words
    see more at :
    https://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization
    https://www.nltk.org/book/ch05.html
    https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words

    PARAMS :
    - 'myList' : the entry list of words
    - 'myStopwordsList' : The list you want to use if you don't want to lemmatize some words

    EXAMPLE :
    smart_lemmatize(['are','a','king','of','snakes','which','comes','from','the','thieves','countries'])
    >>>>>>>>>>>>>>> ['be','a','king','of','snake','which','come','from','the','thief','country']
    """
    from nltk import pos_tag
    from nltk.stem import WordNetLemmatizer

    lemmatized_list = []
    #We use 'pos_tag' ("part of speech tag) in order to regognize the grammatical essence of each word :
    grammatical_essence_list = pos_tag(myList,tagset='universal')
    #We create a dict in order to map the results of the args which comes from the 'pos_tag' processing
    dict_pos_tag_mapping = {'ADJ':'a','ADJ_SAT':'s','ADV':'r','NOUN':'n','VERB':'v'}
    smart_lemmatizer = WordNetLemmatizer()

    #And we associate a smart lemme to each word contained in our list :
    for tuple_ in grammatical_essence_list :
        word = tuple_[0]
        raw_essence = tuple_[1]
        if raw_essence in dict_pos_tag_mapping.keys():
            essence = dict_pos_tag_mapping[raw_essence]
        else:
            essence = 'n'
        if not word in myStopwordsList:
            lemme = smart_lemmatizer.lemmatize(word,essence)
            lemmatized_list.append(lemme)
        else:
            lemmatized_list.append(word)

    return lemmatized_list
