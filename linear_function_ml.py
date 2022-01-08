from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random

class LinearFunction:
    def __init__(self, m, n):
        self.m = float(m)
        self.n = float(n)

    def create_training_data(self, size):
        if size < 1:
            raise ValueError
        else:
            x = np.arange(-size // 2, size // 2 + 1)
            return x, self.m * x + self.n

    def create_testing_data(self, size):
        if size < 1:
            raise ValueError
        else:
            x = np.arange(-size // 2, size // 2 + 1)
            return x, self.m * x + self.n

    def plot(self, start=-10, end=10):
        x = np.arange(start, end+1)
        y = self.m * x + self.n
        plt.figure(1, figsize=(10, 10))
        plt.title(f'Graph of {self}')
        plt.plot(x, y)
        plt.show()

    def __str__(self):
        if self.n < 0:
            return f'y={self.m}X {self.n}'
        return f'y={self.m}X+{self.n}'


model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

my_function = LinearFunction(2, -1)
x, y = my_function.create_training_data(10)

model.fit(x, y, epochs=500)

size = random.randint(1, 100)
x_test, y_test = my_function.create_training_data(size)
y_prediction = model.predict(x_test).reshape(1, size)

plt.figure(figsize=(10, 10))
plt.title(f'{my_function} Graph Machine Learning')
plt.subplot(2, 1, 1)
plt.title('Training')
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(2, 1, 2)
plt.title('Test')
plt.plot(x_test, y_prediction, 'c-', x_test, y_test, 'r:')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Model', 'Given'])
plt.show()
