#loading all the input data
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from sklearn.utils import shuffle

training_file = 'German Traffic Sign Data/train.p'
validation_file = 'German Traffic Sign Data/valid.p'
testing_file = 'German Traffic Sign Data/test.p' 

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, Y_train = train['features'],train['labels']
X_valid, Y_valid = valid['features'],valid['labels']
X_test, Y_test = test['features'],test['labels']
Y_train, Y_valid, Y_test = np.float64(Y_train), np.float64(Y_valid), np.float64(Y_test)

#input data parameters
n_input = len(X_train)
n_classes = np.unique(Y_train).size
n_valid = len(X_valid)
n_test = len(X_test)
input_image_shape = X_train[0].shape

##grayscale and normalising the train images
def normalise(X_train):
    # grayscale
    grayscaled_images = np.sum(X_train/3, axis=3, keepdims=True)
    
    # normalize
    normalized_images = (grayscaled_images - 128) / 128
    
    
    return normalized_images      #range mapping for pixel intensities to lie between (-1,1)

#segregating the images label-wise only for the purpose of visualisation
def labelWise(Y_train,X_train_g,n_classes=43):
    count_list = []
    labelwise_im = []
    for j in range(n_classes):
        labelim = []
        count = 0
        for i in range(len(Y_train)):
            if Y_train[i] == j:
                count = count+1
                labelim.append(X_train_g[i])
        labelwise_im.append(labelim)        
        count_list.append(count)
    return count_list, labelwise_im
        # labels = np.int32(np.linspace(0,42,43))
        # plt.bar(labels,frequency)

#visualising random data (50 random images) from all the different labels
def visualise(labelwise_im,label,n=50):
    k=0
    rand = random.sample(range(len(labelwise_im[label])),n)
    n_columns = 10
    n_rows = 5
    width, height = 10, 10
    fig, a = plt.subplots(n_rows,n_columns,figsize=(width,height))
    for i in range(n_columns):
        for j in range(n_rows):
            a[j,i].axis('off')
            a[j,i].imshow(labelwise_im[label][rand[k]],cmap='gray')
            k = k+1 


#defining the model architecture
def ModelOutput(image_g):
    #weights and bias definition
    mu = 0
    sigma = 0.1
    weights = {'C1': tf.Variable(tf.truncated_normal((5,5,1,6),mean=mu,stddev=sigma)),
                'C2': tf.Variable(tf.truncated_normal((5,5,6,16),mean=mu,stddev=sigma)),
                'C3': tf.Variable(tf.truncated_normal((5,5,16,32),mean=mu,stddev=sigma)),
                'FC1': tf.Variable(tf.truncated_normal((512,120),mean=mu,stddev=sigma)),
                'FC2': tf.Variable(tf.truncated_normal((120,84),mean=mu,stddev=sigma)),
                'FC3': tf.Variable(tf.truncated_normal((84,43),mean=mu,stddev=sigma))}
    bias = {'C1': tf.Variable(tf.zeros(6)),
                'C2': tf.Variable(tf.zeros(16)),
                'C3': tf.Variable(tf.zeros(32)),
                'FC1': tf.Variable(tf.zeros(120)),
                'FC2': tf.Variable(tf.zeros(84)),
                'FC3': tf.Variable(tf.zeros(43))}
    #layer output definitions
    #first convolution layer - 32x32x1 to 28x28x6 & activation
    conv1 = tf.add(tf.nn.conv2d(image_g,weights['C1'],strides=[1,1,1,1],padding='VALID'),bias['C1'])
    conv1 = tf.nn.relu(conv1)
    
    #first max-pooling layer - 28x28x6 to 24x24x16
    conv2 = tf.add(tf.nn.conv2d(conv1,weights['C2'],strides=[1,1,1,1],padding='VALID'),bias['C2'])
    conv2 = tf.nn.relu(conv2)
    
    #first max-pooling layer - 24x24x16 to 12x12x16
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    #second convolution layer - 12x12x16 to 8x8x32
    conv3 = tf.add(tf.nn.conv2d(conv2,weights['C3'],strides=[1,1,1,1],padding='VALID'),bias['C3'])
    conv3 = tf.nn.relu(conv3)
    
    #second max-pooling layer - 10x10x16 to 5x5x16
    conv3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    #flatten the 5x5x16 data to a vector of size (5*5*16) = 400
    conv3 = tf.reshape(conv3,[-1,weights['FC1'].get_shape().as_list()[0]])

    #first fully connected layer & output
    fc1 = tf.add(tf.matmul(conv3,weights['FC1']),bias['FC1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1,keep_prob)
    
    #second fully connected layer & output
    fc2 = tf.add(tf.matmul(fc1,weights['FC2']),bias['FC2'])
    fc2 = tf.nn.relu(fc2)
    
    #second fully connected layer & output
    fc3 = tf.add(tf.matmul(fc2,weights['FC3']),bias['FC3'])
    return fc3

batch_size = 128
epochs = 100
learning_rate= 0.0009

x = tf.placeholder(tf.float32,(None,32,32,1))
y = tf.placeholder(tf.int32,(None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y,43)
logits = ModelOutput(x)
loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y,logits=logits)
avg_loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(avg_loss)
correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(one_hot_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

def ModelAccuracy(X,Y):
    num = len(X)
    total_accuracy = 0
    sess = tf.get_default_session()
    for i in range(0,num,batch_size):
        x_batch, y_batch = X[i:i+batch_size], Y[i:i+batch_size]
        accuracy_batch = sess.run(accuracy, feed_dict={x:x_batch, y:y_batch, keep_prob:1})
        total_accuracy += (accuracy_batch * len(x_batch))
    return total_accuracy/num
        
    
  
#preprocessing the data
X_train, Y_train = shuffle(X_train, Y_train)
X_train_g = normalise(X_train)
X_valid_g = normalise(X_valid)
X_test_g = normalise(X_test)
# frequency, X_train_labelwise = labelWise(Y_train,X_train_g,n_classes=43)
# visualise(X_train_labelwise,label=16,n=50)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_datapoints = len(X_train_g)
    validation_all = []
    for i in range(epochs):
        X_train_g, Y_train = shuffle(X_train_g, Y_train)
        for j in range(0,num_datapoints,batch_size):
            x_batch, y_batch = X_train_g[i:i+batch_size], Y_train[i:i+batch_size]
            sess.run(optimizer, feed_dict={x: x_batch, y: y_batch, keep_prob:0.5})
            
        training_a = ModelAccuracy(X_train_g,Y_train)
        validation_a = ModelAccuracy(X_valid_g,Y_valid)
        validation_all.append(validation_a)
        print("Epoch# : {}".format(i+1))
        print("Training accuracy {}".format(training_a))
        print("Validation accuracy: {}".format(validation_a))
    
    print("Training Completed")
    saver.save(sess, './lenet')
    plt.plot(range(epochs),validation_all)
    plt.show()
    
    #test accuracy
    test_accuracy = ModelAccuracy(X_test_g,Y_test)
    print("Test Accuracy: {}".format(test_accuracy))    


