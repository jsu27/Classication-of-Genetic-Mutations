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
'''
def read_data(file_path):
    fh = open(file_path, 'r')
    data = []
    for line in fh.readLines():
        line = line[line.find('\|\|')+4:] #entire research paper
        for sent in sent_tokenize(line):
            data.append(sent)'''
top_words = 5000
#testx = pd.read_csv('Data/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train, test = train_test_split(trainx, test_size=0.3)
X_train1 = list(train["Text"])#[nltk.word_tokenize(str(text)) for text in train["Text"]]
y_train1 = list(train["Class"])
X_train = []
y_train = []
for i in range(train.shape[0]):
    text = X_train1[i]
    for sent in sent_tokenize(text):
        X_train.append(sent)
        y_train.append(y_train1[i])
print(X_train[:10])
print(y_train[:10])
y_train = to_categorical(y_train, num_classes=10) #classes 1-9

X_test1 = list(test["Text"]) #[nltk.word_tokenize(str(text)) for text in test["Text"]]
y_test1 = list(test["Class"])
X_test = []
y_test = []
for i in range(test.shape[0]):
    text = X_test1[i]
    for sent in sent_tokenize(text):
        X_test.append(sent)
        y_test.append(y_test1[i])
print(X_test[:10])
print(y_test[:10])
y_test = to_categorical(y_test, num_classes=10)

print("X_train len: {}".format(len(X_train)))
print("y_train len: {}".format(len(y_train)))
print("X_test len: {}".format(len(X_test)))
print("y_test len: {}".format(len(y_test)))
print("Split train and test data.")

# truncate and pad input sequences
t = Tokenizer()
t.fit_on_texts(X_train) #tokenizes each word; integer encodes words
vocab_size = len(t.word_index) + 1 #vocab_size; size of vocab of training text = 256994

# encode the documents
encoded_train = t.texts_to_sequences(X_train)
#max_text_length = np.median([len(text) for text in encoded_train])
max_text_length = 700 #7356 #from calculations
#max_text_length = int(max_text_length.round())
#print("Avg text length: {}".format(max_text_length))
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
model.add(Bidirectional(LSTM(embedding_vector_length)))
model.add((Dense(10, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())
model.fit(X_train[:10000], y_train[:10000], validation_data=(X_test[:3000], y_test[:3000]), epochs=10, batch_size=100)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
