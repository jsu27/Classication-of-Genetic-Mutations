import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
#from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import data_retrieval as dr
# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
train_variants = pd.read_csv('Data/training_variants', nrows=top_words)
trainx = pd.read_csv('Data/training_text', sep="\|\|", engine='python', header=None, skiprows=1, nrows=top_words, names=["ID","Text"])

#testx = pd.read_csv('Data/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train, test = train_test_split(trainx, test_size=0.3)

# truncate and pad input sequences
max_text_length = 250 #???

t = Tokenizer()
t.fit_on_texts(dr.SAMPLES)
vocab_size = len(t.word_index) + 1 #dr.vocab_size; size of vocab of training text = 256994

# integer encode the documents
#encoded_docs = t.texts_to_sequences()
#print(encoded_docs)
# pad documents to a max length of 4 words
X_train = sequence.pad_sequences(X_train, maxlen=max_text_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_text_length)
#WORD EMBEDDINGS
embeddings_index = dict()
f = open('word_vectors/glove.6B/glove.6B.50d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))

#EMBED WORDS IN DATA (USE GLOVE)
embedding_matrix = zeros((vocab_size, 50))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


#EMBED IN MODEL
embedding_vector_length = 50
model = Sequential()
e = Embedding(vocab_size, embedding_vector_length, weights=[embedding_matrix], input_length=max_review_length, trainable=False) #input_dim (vocab size), vector dim, input_len (# words per document)
model.add(e)
#model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length)) #input_dim (vocab size), vector dim, input_len (# words per document)
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)




# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))