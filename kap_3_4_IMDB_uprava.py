from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) #num_words=10000 ponecháváme si 10 000 nejčastějších slov

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])  #binary_crossentropy

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
#print(results)
print(model.predict(x_test))

# pro 2 skryté vrsvty(16 units) a binary_crossentropy je pravděpodobnost:
# [[0.19341272]
#  [0.99985063]
#  [0.7743441 ]
#  ...
#  [0.10543334]
#  [0.07295531]
#  [0.38703653]]

# pro 3 skryté vrsvty(16 units) a binary_crossentropy je pravděpodobnost:
# [[0.18478304]
#  [0.9999268 ]
#  [0.9296993 ]
#  ...
#  [0.13693342]
#  [0.07997286]
#  [0.71858513]]

# pro 3 skryté vrsvty(32,32,64 units) a binary_crossentropy je pravděpodobnost:
# [[0.14691755]
#  [0.9997062 ]
#  [0.9802506 ]
#  ...
#  [0.22334826]
#  [0.08415356]
#  [0.8023231 ]]

# pro 3 skryté vrsvty(64 units) a mse loss fce je pravděpodobnost:
# [[0.06527615]
#  [0.9996803 ]
#  [0.6451663 ]
#  ...
#  [0.12785086]
#  [0.07635078]
#  [0.8536216 ]]