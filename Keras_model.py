import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, TimeDistributed, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
# fix random seed for reproducibility
np.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
train_variants = pd.read_csv('Data/training_variants', nrows=top_words)
trainx = pd.read_csv('Data/training_text', sep="\|\|", engine='python', dtype={'Text': str}, header=None, skiprows=1, nrows=top_words, names=["ID","Text"])
trainx["Class"] = train_variants["Class"]
print("Loaded data.")

X1 = list(trainx["Text"])
y1 = list(trainx["Class"])
X = []
y = []
for i in range(trainx.shape[0]):
    text = X1[i]
    for sent in sent_tokenize(text):
        X.append(sent)
        y.append(y1[i])
print(X[:10])
print(y[:10])
y = to_categorical(y, num_classes=10) #classes 1-9
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)

print("X_train len: {}".format(len(X_train))) #965658
print("y_train len: {}".format(len(y_train)))
print("X_test len: {}".format(len(X_test))) #420876
print("y_test len: {}".format(len(y_test)))
print("Split train and test data.")

# truncate and pad input sequences
t = Tokenizer()
t.fit_on_texts(X_train) #tokenizes each word; integer encodes words
vocab_size = len(t.word_index) + 1 #vocab_size; size of vocab of training text = 256994

# encode the documents
encoded_train = t.texts_to_sequences(X_train)
print(encoded_train[:10])
max_text_length = int(np.median([len(text) for text in encoded_train]))
print(max_text_length)
#max_text_length = 22 #from calculations

encoded_test = t.texts_to_sequences(X_test)

# pad documents to a max length
X_train = sequence.pad_sequences(encoded_train, maxlen=max_text_length)
X_test = sequence.pad_sequences(encoded_test, maxlen=max_text_length)
print("Padded data.")
#WORD EMBEDDINGS
embeddings_index = dict()
f = open('word_vectors/glove.6B/glove.6B.50d.txt', encoding='utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Loaded {} word vectors.'.format(len(embeddings_index.keys())))

#EMBED WORDS IN DATA (USE GLOVE)
embedding_vector_length = 50
embedding_matrix = np.zeros((vocab_size, embedding_vector_length))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

#CREATE RNN MODEL
model = Sequential()
e = Embedding(vocab_size, embedding_vector_length, weights=[embedding_matrix], input_length=max_text_length, trainable=True) #input_dim (vocab size), vector dim, input_len (# words per document)
model.add(e)
model.add(Bidirectional(LSTM(embedding_vector_length, return_sequences = True)))
model.add(Bidirectional(LSTM(embedding_vector_length, return_sequences = True)))
model.add(Bidirectional(LSTM(embedding_vector_length)))
model.add((Dense(10, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=100)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
