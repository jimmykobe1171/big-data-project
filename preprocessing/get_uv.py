

import csv
import pickle
import numpy as np
from numpy import mean
from scipy.sparse import csr_matrix

import gensim
from textblob import TextBlob

def main():

    fname = 'bigram_model.bin'
    model = gensim.models.Word2Vec.load(fname)

    review_csv = 'reviews.csv'
    
    f = open(review_csv, 'r')
    lines = csv.reader(f)

    u_dict = {}
    r_dict = {}

    m = 5000
    n = 26730

    u_list = range(m)
    r_list = range(n)

    for line_list in lines:
        uid = int(line_list[0])
        rid = int(line_list[1])
        review = line_list[2]

        if uid not in u_dict:
            u_dict[uid] = [review]
        else:
            u_dict[uid].append(review)

        if rid not in r_dict:
            r_dict[rid] = [review]
        else:
            r_dict[rid].append(review)

    f.close()

    # Build User prime vectors
    U_prime_vector = []
    for uid in u_list:
        sentences_number = 1
        u_aspect_vector = [0, 0, 0, 0, 0]

        if uid in u_dict:
            review_list = u_dict[uid]

            for review in review_list:
                sentence_list = review.split('.')

                for sentence in sentence_list:
                    words = sentence.split()

                    if words:
                        
                        aspect_index = get_aspect(model, words)

                        if aspect_index == -1:
                            continue

                        sentences_number += 1

                        # Find an aspect --> user --> no need to do sentiment
                        u_aspect_vector[aspect_index] += 1

        u_vector = [num / sentences_number for num in u_aspect_vector]

        U_prime_vector.append(u_vector)

    # Save the list using pickle
    # tmp = np.array(U_prime_vector, dtype=np.float)
    tmp = csr_matrix(U_prime_vector, dtype=np.float)
    print(tmp.shape)
    pickle.dump(tmp, open( "U_prime_vector.p", "wb" ) )

    R_prime_vector = []
    for rid in r_list:
        r_aspect_vector = [[0.0], [0.0], [0.0], [0.0], [0.0]]

        if rid in r_dict:
            review_list = r_dict[rid]

            for review in review_list:
                sentence_list = review.split('.')

                for sentence in sentence_list:
                    words = sentence.split()

                    if words:
                        aspect_index = get_aspect(model, words)

                        if aspect_index == -1:
                            continue

                        # Find an aspect --> do sentiment analysis
                        t = TextBlob(sentence)
                        r_aspect_vector[aspect_index].append(t.sentiment.polarity)

        r_vector = [mean(l) for l in r_aspect_vector]

        R_prime_vector.append(r_vector)

    # Save using pickle
    # tmp = np.array(R_prime_vector, dtype=np.float)

    tmp = csr_matrix(R_prime_vector, dtype=np.float)
    print(tmp.shape)
    pickle.dump(tmp, open( "R_prime_vector.p", "wb" ))

def get_aspect(model, words):

    aspect_list = [['foods'], ['severs'], ['ambience'], ['pricy']]

    max_similarity = 0.0
    aspect_index = -1
    for i, aspect in enumerate(aspect_list):
        try:
            similaity = model.n_similarity(words, aspect)
        except KeyError:
            similaity = 0.0

        if similaity > max_similarity and similaity > 0.3:
            max_similarity = similaity
            aspect_index = i

    return aspect_index

if __name__ == '__main__':
    main()









