def baseline_synset_vector(word_vectors, words, k):
    return word_vectors.most_similar(positive=words, topn=k)  # [(word, similarity)]