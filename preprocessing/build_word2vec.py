
import gensim

from get_sentences import Sentences

data_dir = './../data/'

raw_data_file = data_dir + 'reviews.txt'

sentences = Sentences(data_dir + raw_data_file)

bigram_transformer = gensim.models.Phrases(sentences)

# model = gensim.models.Word2Vec(sentences)
model = gensim.models.Word2Vec(bigram_transformer[sentences], size=100)

model.save('bigram_model.bin')