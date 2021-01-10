#Klasifikace filmových recenzí, 25 tis. pro tréning, 25 tis. pro testování, rozděleny 50% kladných, 50 záporných
from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) #num_words=10000 ponecháváme si 10 000 nejčastějších slov
# train_labels a test_labels -> list 0 nebo 1: 0 = negativní, 1 = pozitivní recenze
# train_data a test_labes -> list recenzí

#train_data[0]
#train_labels[0]

# word_index -> slovník mapující slova na index
# word_index = imdb.get_word_index()
# #otočení mapování - z int na slova
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# # indexy jsou posunuty o 3
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
#print(decoded_review)

def vectorize_sequences(sequences, dimension=10000):
    # Vytvoří nulovou matici o velkosti (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # nastaví i-tý index v result na 1
    return results

# Vektorizované tréningová data
x_train = vectorize_sequences(train_data)
# Vektorizované testovací data
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) # --| Dvě vrstvy s dimenzí 16, obě používají relu usměrňovač (https://cs.qaz.wiki/wiki/Rectifier_(neural_networks))
model.add(layers.Dense(16, activation='relu'))                       # --|
model.add(layers.Dense(1, activation='sigmoid')) # S křivka -> výstup je skóre mezi 0 a 1

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# můžu předat objekty funkcí optimizer, ztrátovou funkci a metriky jako argumenty.
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])

#--------------------Ověření postupu---------------------------------------------
# Ověřovací set dat o velikosti 10 000
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
# Trénování modelu
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
# data jsou rozdělena do 20 epoch po 512 recenzích
# výpis slovníku, který obsahuje data co se v průběhu trénování stalo
history_dict = history.history
print(history_dict.keys())
# dict_keys(['val_acc', 'acc', 'val_loss', 'loss'])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
#-------------------------------------------------------------
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
#-------------------------------------------------------------
plt.clf()   # clear figure
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
#-------------------------------------------------------------
# training loss se s každou epochou snižuje a training acc se zvyšuje.
# pro validation loss a acc je peak ve 4 epoše -> overfitting, model se chová dobře na trénovacích datech, ale ne na validačních
