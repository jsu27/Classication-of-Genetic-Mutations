import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
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
DEBUG = False
# load the datasets: trainx = text data; train_variants = classes
trainx = pd.read_csv('Data/training_text', sep="\|\|", engine='python', dtype={'Text': str}, header=None, skiprows=1, names=["ID","Text"])
train_variants = pd.read_csv('Data/training_variants')
print("Loaded data.")

#Create X, y data
X1 = list(trainx["Text"])
y1 = list(train_variants["Class"]) #[568, 452, 89, 686, 242, 275, 953, 19, 37]
X = []
y = []

#tokenize by sentence
for i in range(trainx.shape[0]):
    text = X1[-1]
    for sent in sent_tokenize(text):
        X.append([sent])
        y.append(y1[-1] - 1) #classes 0 - 8 instead of 1 - 9
    del X1[-1]
    del y1[-1]

#class_count = [235547, 182953, 24939, 271578, 76715, 79296, 485804, 9534, 20168]
class_counts = [0] * 9
for i in range(len(y1)):
    num = y1[i] - 1
    class_counts[num] += 1
print(class_counts)

ratio = 1 #ratio of data
if (DEBUG):
    ratio = 0.7

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7*ratio, test_size=0.3*ratio, shuffle=True)

print("X_train len: {}".format(len(X_train))) #965658
print("y_train len: {}".format(len(y_train)))
print("X_test len: {}".format(len(X_test))) #420876
print("y_test len: {}".format(len(y_test)))
print("Split train and test data.")

rus = RandomUnderSampler(random_state=0)
X_train, y_train = rus.fit_sample(X_train, y_train)
y_train = to_categorical(y_train, num_classes=9) #one hot vector
print(X_train[:10])
print("X train resampled len", len(X_train))
print(y_train[:10])
print("y train resampled len", len(y_train))

X_test, y_test = rus.fit_sample(X_test, y_test)
y_test = to_categorical(y_test, num_classes=9) #one hot vector
print(X_test[:10])
print("X test resampled len", len(X_test))
print(y_test[:10])
print("y test resampled len", len(y_test))
X_train = [X_train[i][0] for i in range(len(X_train))] #6761 per class
X_test = [X_test[i][0] for i in range(len(X_test))] #2773 per class

def class_count(y_data, n): #find # of samples per class
    class_counts = [0] * n
    for i in range(len(y_data)):
        num = np.argmax(y_data[i])
        class_counts[num] += 1
    return class_counts
print("train classes: {}".format(class_count(y_train, 9)))
print("test classes: {}".format(class_count(y_test, 9)))

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

#import word embeddings
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

#embed words
embedding_vector_length = 50
embedding_matrix = np.zeros((vocab_size, embedding_vector_length))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None):
        embedding_matrix[i] = embedding_vector
    else: #if word not found, create random vector
        embedding_matrix[i] = np.random.uniform(-1, 1, embedding_vector_length)
dropout = 0.55
epochs = 150
if DEBUG:
    epochs = 25
rand_search_dropout = True
rand_search_lr = True
lr = 0.0001
lrs = [0.00003, 0.00005]#[.001, 0.0005, 0.0001]#[0.00003, 0.00005, 0.00001]#[0.00001, 0.00003, 0.000007]#[0.00007, 0.0001, 0.00005]#[0.00001, 0.00003, 0.00005, 0.00007, 0.0001, 0.0003]#[0.00000001, 0.0000001, 0.000001, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]
#dropouts = [0.2, 0.25, 0.3, 0.35, 0.4]#[0.35, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75]
if (rand_search_lr):
    for i in range(len(lrs)):
        lr = lrs[i]
        print("lr:", lr)
        #CREATE RNN MODEL
        model = Sequential()
        e = Embedding(vocab_size, embedding_vector_length, weights=[embedding_matrix], input_length=max_text_length, trainable=True)
        model.add(e)
        model.add(Bidirectional(LSTM(embedding_vector_length, return_sequences = True)))
        model.add(Bidirectional(LSTM(embedding_vector_length, return_sequences = True)))
        model.add(Bidirectional(LSTM(embedding_vector_length)))
        model.add(Dropout(dropout)) #[0, 1]
        model.add(Dense(9, activation='softmax'))
        adam = keras.optimizers.Adam(lr=lr) #[e-10, 1]
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        print(model.summary())
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=100)

        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: {}%".format(round(scores[1]*100, 2)))

        #make predictions
        y_pred = model.predict(X_test)
        y_pred = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
        y_test2 = [np.argmax(y_test[i]) for i in range(len(y_test))]

        #metrics: accuracy, precision, recall
        print("Accuracy:", accuracy_score(y_test2, y_pred))
        print("Precision (weighted):", precision_score(y_test2, y_pred, average='weighted'))
        print("Precision (micro):", precision_score(y_test2, y_pred, average='micro'))
        print("Recall (weighted):", recall_score(y_test2, y_pred, average='weighted'))
        print("Recall (micro):", recall_score(y_test2, y_pred, average='micro'))

        #confusion matrix
        cm = confusion_matrix(y_test2, y_pred)
        print(cm)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)

        # plot accuracy, loss, conf_mat
        plt.figure(3*i+1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy: lr={0} acc={1}'.format(lr, round(scores[1]*100, 2)))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('acc{0}.png'.format(i))
        # summarize history for loss
        plt.figure(3*i+2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss: lr={}'.format(lr))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss{}.png'.format(i))

        plt.clf()
        plt.matshow(cm, fignum=False)
        plt.title('Confusion matrix: lr={}'.format(lr))
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('cm{}.png'.format(i))
