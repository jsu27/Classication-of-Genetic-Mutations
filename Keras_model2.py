import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, TimeDistributed, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import keras.optimizers
# fix random seed for reproducibility
np.random.seed(7)
DEBUG = True
# load the datasets: trainx = text data; train_variants = classes
trainx = pd.read_csv('Data/training_text', sep="\|\|", engine='python', dtype={'Text': str}, header=None, skiprows=1, names=["ID","Text"])
train_variants = pd.read_csv('Data/training_variants')
print("Loaded data.")

#Create X, y data
X1 = list(trainx["Text"])
y1 = list(train_variants["Class"])
X = []
y = []
#tokenize by sentence
for i in range(trainx.shape[0]):
    text = X1[i]
    for sent in sent_tokenize(text):
        X.append(sent)
        y.append(y1[i] - 1) #classes 0 - 8 instead of 1 - 9
y = to_categorical(y, num_classes=9) #one hot vector

ratio = 0.1 #ratio of data
if (DEBUG):
    ratio = 0.01
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7*ratio, test_size=0.3*ratio)

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
#max_text_length = int(np.median([len(text) for text in encoded_train]))
#max_text_length = int(np.average([len(text) for text in encoded_train]))
#print(max_text_length)
max_text_length = 24 #med = 22, avg=24
encoded_test = t.texts_to_sequences(X_test)

# pad documents to a max length
X_train = sequence.pad_sequences(encoded_train, maxlen=max_text_length)
X_test = sequence.pad_sequences(encoded_test, maxlen=max_text_length)
print("Padded data.")

#WORD EMBEDDINGS
embeddings_index = dict()
f = open('word_vectors/glove.6B/glove.6B.50d.txt', encoding='utf-8')
#f = open('word_vectors/pub.50.vec/pub.50.vec', encoding='latin-1') #pubmed data
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:])
	embeddings_index[word] = coefs
f.close()

print('Loaded {} word vectors.'.format(len(embeddings_index.keys())))

#EMBED WORDS FOR DATA
embedding_vector_length = 50
embedding_matrix = np.zeros((vocab_size, embedding_vector_length))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None):
        embedding_matrix[i] = embedding_vector
    else: #if word not found, create random vector
        embedding_matrix[i] = np.random.uniform(-1, 1, embedding_vector_length)
dropout = 0.7
epochs = 3
rand_search_dropout = True
lr = 0.0001

if (rand_search_dropout):
    for i in range(10):
        dropout = round(np.random.uniform(0, 1), 2)
        #CREATE RNN MODEL
        model = Sequential()
        e = Embedding(vocab_size, embedding_vector_length, weights=[embedding_matrix], input_length=max_text_length, trainable=True)
        model.add(e)
        model.add(Bidirectional(LSTM(embedding_vector_length, return_sequences = True)))
        model.add(Bidirectional(LSTM(embedding_vector_length, return_sequences = True)))
        model.add(Bidirectional(LSTM(embedding_vector_length)))
        model.add(Dropout(dropout)) #[0, 1]
        model.add((Dense(9, activation='softmax')))
        adam = keras.optimizers.Adam(lr=lr) #[e-10, 1]
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        print(model.summary())
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: {}%" % (round(scores[1]*100, 2)))

        y_pred = model.predict(X_test)
        y_pred = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
        y_test2 = [np.argmax(y_test[i]) for i in range(len(y_test))]
        cm = confusion_matrix(y_test2, y_pred)
        print(cm)

        # summarize history for accuracy
        plt.figure(3*i+1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy: dropout={0} acc={1}'.format(dropout, round(scores[1]*100, 2)))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('acc{0}.png'.format(i))
        # summarize history for loss
        plt.figure(3*i+2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss: dropout={}'.format(dropout))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss{}.png'.format(i))

        plt.clf()
        plt.matshow(cm, fignum=False)
        plt.title('Confusion matrix: dropout={}'.format(dropout))
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('cm{}.png'.format(i))

        #plt.show()
