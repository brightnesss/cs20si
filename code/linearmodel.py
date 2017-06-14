"""
Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from
the number of file in the city of Chicago
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

# Step 0: huber loss


def huber_loss(labels, predicted_labels, delta=1.0):
    residual = tf.abs(predicted_labels - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res, 'loss')


DATA_FILE = '../data/fire_theft.xls'

# Step 1: read data from .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_sample = sheet.nrows - 1

# Step 2: creat placeholders for input X and label Y
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: creat weight and bias, initialized to 0
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# Step 4: build model to predict Y
Y_predicted = X * w + b

# Step 5: use the square error as the loss function
#loss = tf.square(Y - Y_predicted, name='loss')
loss = huber_loss(Y, Y_predicted)

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.001).minimize(loss)

sess = tf.Session()

# Step 7: initialize the necessary variables, in this case, w and b
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('../graph', sess.graph)

# Step 8: train the model
for i in range(100):  # train the model 100 times
    total_loss = 0
    for x, y in data:
        # Session runs trian_op and fetch values of loss
        _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
        total_loss += l
    print('Epoch {0}: {1}'.format(i, total_loss / n_sample))

# close the writer when you're done using it
writer.close()

# Step 9: output the values of w and b
w_value, b_value = sess.run([w, b])

sess.close()

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real Data')
plt.plot(X, X * w_value + b_value, 'r', label='Predicted Data')
plt.legend()
plt.show()
