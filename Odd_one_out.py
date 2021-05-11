import gensim
from gensim.models import word2vec , KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

word_vector = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary= True)

def odd_one_out(words):

    all_word_vector = [word_vector[w] for w in words]
    average_vector = np.mean(all_word_vector, axis = 0)
    odd_one_out = None
    min_cosine_similarity = 1.0
    for i in words:
        sim = cosine_similarity([word_vector[i]],[average_vector])
        if sim < min_cosine_similarity:
            min_cosine_similarity = sim
            odd_one_out = i
        print(f'Cosine Similarity between {i} & average vector is {sim}')

    return odd_one_out
