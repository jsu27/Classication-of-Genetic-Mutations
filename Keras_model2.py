import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, TimeDistributed, Bidirectional, Dropout
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
        y.append(y1[i] - 1) #CLASSES: 0 - 8 now
print(X[:10])
print(y[:10])
y = to_categorical(y, num_classes=9) #classes 0 - 8
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7*0.3, test_size=0.3*0.3)

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
#max_text_length = int(np.median([len(text) for text in encoded_train]))
#print(max_text_length)
max_text_length = 40 #from calculations

encoded_test = t.texts_to_sequences(X_test)

# pad documents to a max length
X_train = sequence.pad_sequences(encoded_train, maxlen=max_text_length)
X_test = sequence.pad_sequences(encoded_test, maxlen=max_text_length)
print("Padded data.")
#WORD EMBEDDINGS
embeddings_index = dict()
#f = open('word_vectors/glove.6B/glove.6B.50d.txt', encoding='utf-8')
f = open('word_vectors/pub.50.vec/pub.50.vec', encoding='latin-1') #pubmed data
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:])
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
model.add(Dropout(0.4))
model.add((Dense(9, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=100)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

print(history.history.keys())
# summarize history for accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()
'''
GLOVE
Train on 291172 samples, validate on 124789 samples
Epoch 1/6
291172/291172 [==============================] - 886s 3ms/step - loss: 1.4658 - acc: 0.4650 - val_loss: 1.2942 - val_acc: 0.5238
Epoch 2/6
291172/291172 [==============================] - 790s 3ms/step - loss: 1.1980 - acc: 0.5631 - val_loss: 1.2000 - val_acc: 0.5593
Epoch 3/6
291172/291172 [==============================] - 796s 3ms/step - loss: 1.0669 - acc: 0.6092 - val_loss: 1.1886 - val_acc: 0.5674
Epoch 4/6
291172/291172 [==============================] - 809s 3ms/step - loss: 0.9860 - acc: 0.6355 - val_loss: 1.1998 - val_acc: 0.5681
Epoch 5/6
291172/291172 [==============================] - 801s 3ms/step - loss: 0.9260 - acc: 0.6542 - val_loss: 1.2124 - val_acc: 0.5726
Epoch 6/6
291172/291172 [==============================] - 794s 3ms/step - loss: 0.8804 - acc: 0.6671 - val_loss: 1.2564 - val_acc: 0.5729
Accuracy: 57.29%
dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

PUBMED
Train on 291172 samples, validate on 124789 samples
Epoch 1/5
291172/291172 [==============================] - 907s 3ms/step - loss: 1.4995 - acc: 0.4484 - val_loss: 1.3592 - val_acc: 0.4975
Epoch 2/5
291172/291172 [==============================] - 918s 3ms/step - loss: 1.2804 - acc: 0.5304 - val_loss: 1.2511 - val_acc: 0.5387
Epoch 3/5
291172/291172 [==============================] - 872s 3ms/step - loss: 1.1553 - acc: 0.5757 - val_loss: 1.2126 - val_acc: 0.5541
Epoch 4/5
291172/291172 [==============================] - 891s 3ms/step - loss: 1.0771 - acc: 0.6023 - val_loss: 1.2006 - val_acc: 0.5590
Epoch 5/5
291172/291172 [==============================] - 921s 3ms/step - loss: 1.0213 - acc: 0.6189 - val_loss: 1.2016 - val_acc: 0.5663
Accuracy: 56.63%

with 50%?? of data, 20 epochs batch_size=100
Train on 485286 samples, validate on 207981 samples
Epoch 1/20
485286/485286 [==============================] - 1472s 3ms/step - loss: 1.3633 - acc: 0.5006 - val_loss: 1.2008 - val_acc: 0.5602
Epoch 2/20
485286/485286 [==============================] - 1400s 3ms/step - loss: 1.1015 - acc: 0.5926 - val_loss: 1.1376 - val_acc: 0.5793
Epoch 3/20
485286/485286 [==============================] - 1528s 3ms/step - loss: 0.9987 - acc: 0.6254 - val_loss: 1.1157 - val_acc: 0.5850
Epoch 4/20
485286/485286 [==============================] - 1556s 3ms/step - loss: 0.9324 - acc: 0.6455 - val_loss: 1.1200 - val_acc: 0.5848
Epoch 5/20
485286/485286 [==============================] - 1398s 3ms/step - loss: 0.8823 - acc: 0.6610 - val_loss: 1.1195 - val_acc: 0.5927
Epoch 6/20
485286/485286 [==============================] - 1671s 3ms/step - loss: 0.8402 - acc: 0.6734 - val_loss: 1.1380 - val_acc: 0.5944
Epoch 7/20
485286/485286 [==============================] - 1808s 4ms/step - loss: 0.8045 - acc: 0.6845 - val_loss: 1.1482 - val_acc: 0.5969
Epoch 8/20
485286/485286 [==============================] - 1565s 3ms/step - loss: 0.7737 - acc: 0.6934 - val_loss: 1.1835 - val_acc: 0.5980
Epoch 9/20
485286/485286 [==============================] - 1630s 3ms/step - loss: 0.7466 - acc: 0.7021 - val_loss: 1.2187 - val_acc: 0.5933
Epoch 10/20
485286/485286 [==============================] - 1619s 3ms/step - loss: 0.7227 - acc: 0.7094 - val_loss: 1.2506 - val_acc: 0.5931
Epoch 11/20
485286/485286 [==============================] - 1618s 3ms/step - loss: 0.7019 - acc: 0.7159 - val_loss: 1.2852 - val_acc: 0.5962
Epoch 12/20
485286/485286 [==============================] - 1649s 3ms/step - loss: 0.6829 - acc: 0.7221 - val_loss: 1.3069 - val_acc: 0.5932
Epoch 13/20
485286/485286 [==============================] - 1709s 4ms/step - loss: 0.6667 - acc: 0.7277 - val_loss: 1.3317 - val_acc: 0.5941
Epoch 14/20
485286/485286 [==============================] - 1661s 3ms/step - loss: 0.6521 - acc: 0.7324 - val_loss: 1.3880 - val_acc: 0.5924
Epoch 15/20
485286/485286 [==============================] - 1602s 3ms/step - loss: 0.6404 - acc: 0.7364 - val_loss: 1.3830 - val_acc: 0.5936
Epoch 16/20
485286/485286 [==============================] - 1473s 3ms/step - loss: 0.6287 - acc: 0.7396 - val_loss: 1.4019 - val_acc: 0.5937
Epoch 17/20
485286/485286 [==============================] - 1301s 3ms/step - loss: 0.6193 - acc: 0.7435 - val_loss: 1.4300 - val_acc: 0.5898
Epoch 18/20
485286/485286 [==============================] - 1301s 3ms/step - loss: 0.6108 - acc: 0.7461 - val_loss: 1.4699 - val_acc: 0.5897
Epoch 19/20
485286/485286 [==============================] - 1324s 3ms/step - loss: 0.6017 - acc: 0.7490 - val_loss: 1.5078 - val_acc: 0.5893
Epoch 20/20
485286/485286 [==============================] - 1605s 3ms/step - loss: 0.5948 - acc: 0.7510 - val_loss: 1.5367 - val_acc: 0.5892
Accuracy: 58.92%
'''
