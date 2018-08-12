def find_similar_neighbors(similarity_matrix, list_of_terms, number_of_neighbors):
    import numpy as np
    L_entities = list(similarity_matrix.index)
    L_indexes = list(range(0,len(L_entities)))
    dict_index_entity = {}

    for index, entity in zip(L_indexes,L_entities):
        dict_index_entity[index] = entity

    dict_neighbors = {}

    for term in list_of_terms:
        L_neighbors = []
        similar_terms = list(similarity_matrix.loc[term,])
        L_neigbors_indexes = sorted(range(len(similar_terms)), key=lambda i: similar_terms[i], reverse=True)[:number_of_neighbors+1]

        for index in L_neigbors_indexes:
            entity = dict_index_entity[index]
            if entity!=term:
                L_neighbors.append(entity)
        dict_neighbors[term] = L_neighbors

    return(dict_neighbors)

def generate_tags(myText,
                  lda_word_sim_matrix,
                  lda_tag_sim_matrix,
                  w2v_word_sim_matrix,
                  w2v_tag_sim_matrix,
                  w2v_pure_tag_sim_matrix,
                  tags,
                  NEIGHBORS,
                  recommendation_filter,
                  THRESHOLD):

    from collections import Counter
    import pandas as pd

    L_recommendation = []
    L_pure_tags_recommendation = []
    L_sim_matrices = [lda_word_sim_matrix,lda_tag_sim_matrix,w2v_word_sim_matrix,w2v_tag_sim_matrix,w2v_pure_tag_sim_matrix]
    L_labels = ['lda_word_sim_matrix','lda_tag_sim_matrix','w2v_word_sim_matrix','w2v_tag_sim_matrix','w2v_pure_tag_sim_matrix']

    corpus = myText.split(" ")
    matching_tags = set(corpus).intersection(set(tags))
    for word in corpus:
        for sim_matrix,label in zip(L_sim_matrices,L_labels):
            if word in sim_matrix.index :
                matrix_recommendation = find_similar_neighbors(sim_matrix,[word],NEIGHBORS)
                matrix_recommendation = matrix_recommendation[word]
                matrix_recommendation = list(set(matrix_recommendation).intersection(set(recommendation_filter)))
                if label != 'w2v_pure_tag_sim_matrix':
                    L_recommendation+= matrix_recommendation
                else:
                    L_pure_tags_recommendation+= matrix_recommendation
                    L_pure_tags_recommendation = sorted(L_pure_tags_recommendation)

            else:
                matrix_recommendation = []
                if label != 'w2v_pure_tag_sim_matrix':
                    L_recommendation+= matrix_recommendation
                else:
                    L_pure_tags_recommendation+= matrix_recommendation
                    L_pure_tags_recommendation = sorted(L_pure_tags_recommendation)

    all_models_counter = dict(Counter(L_recommendation))
    pure_tags_counter = dict(Counter(L_pure_tags_recommendation))
    df_all_models_counter = pd.DataFrame(all_models_counter,index=['occurences']).T
    df_pure_tags_counter = pd.DataFrame(pure_tags_counter,index=['occurences']).T
    df_pure_tags_counter['occurences'] = 4*df_pure_tags_counter['occurences']

    df_final_counter = df_all_models_counter.append(df_pure_tags_counter)
    df_final_counter = df_final_counter.groupby(df_final_counter.index).sum()
    df_final_counter.sort_values(by='occurences', ascending=False, inplace=True)

    quantiles = df_final_counter['occurences'].quantile(q=[THRESHOLD])
    quantiles = quantiles.to_dict()

    df_final_counter = df_final_counter[df_final_counter['occurences']>=quantiles[THRESHOLD]]
    L_final_recommendation = set(list(df_final_counter.index)+list(matching_tags))

    return(L_final_recommendation)
