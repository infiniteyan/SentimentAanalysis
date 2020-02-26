import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from pandas.core.frame import DataFrame

MAX_SIZE = 34
embedding_size = 300

# make test data
test = pd.read_csv('test.csv')['Tweet']
lines = []
for sentence in test:
    sentence = sentence.strip('!?_')
    sentence = sentence.strip()
    words = sentence.split()
    line = []
    for word in words:
        aword = word.strip('!?_')
        if aword:
            lists = aword.split('\'')
            if len(lists) == 2:
                if lists[0]:
                    line.append(lists[0])
                if lists[1]:
                    line.append(lists[1])
            else:
                line.append(aword)
    lines.append(line)
ebeddings = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary = True)
testData = []
testSteps = []
test_negwords = []
for line in lines:
    wordList = []
    v_count = 0
    inv_count = 0
    for i in range(MAX_SIZE):
        if i < len(line):
            try:
                wordList.append(ebeddings[line[i]])
                v_count += 1
            except KeyError:
                test_negwords.append(line[i])
                inv_count += 1
        else:
            wordList.append(np.array([0.0] * embedding_size))
    for j in range(inv_count):
        wordList.append(np.array([0.0] * embedding_size))
    testData.append(wordList)
    testSteps.append(v_count)
testData = np.array(testData)
testSteps = np.array(testSteps)

data = pd.read_csv('train.csv')
print data.shape
target_df = data.loc[:, ['class_label', 'tokens']]
target_df['maxlen'] = target_df['tokens'].apply(lambda x:len(x.strip('[]').split(',')))

# make train features
df = target_df.values
lineList = []
lineSteps = []
train_negwords = []
for line in df:
    words = line[1].strip('[]').split(',')
    wordList = []
    valid_count = 0
    invalid_count = 0
    for i in range(MAX_SIZE):
        if i < len(words):
            word = eval(words[i])
            try:
                wordList.append(ebeddings[word])
                valid_count += 1
            except KeyError:
                train_negwords.append(word)
                invalid_count += 1
        else:
            wordList.append(np.array([0.0] * embedding_size))
    for j in range(invalid_count):
        wordList.append(np.array([0.0] * embedding_size))
    print valid_count, '-----------------------', invalid_count
    lineList.append(wordList)
    lineSteps.append(valid_count)

lineList = np.array(lineList)
lineSteps = np.array(lineSteps)

#make train labels
lineLabels = []
for line in df:
    if line[0] == 1:
        lineLabels.append([1, 0])
    else:
        lineLabels.append([0, 1])

num_nodes = 256
batch_size = 15
output_size = 2

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, MAX_SIZE, embedding_size))
    tf_train_steps = tf.placeholder(tf.int32, shape = (batch_size))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, output_size))

    tf_test_dataset = tf.constant(testData, tf.float32)
    tf_test_steps = tf.constant(testSteps, tf.int32)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = num_nodes, state_is_tuple = True)
    w1 = tf.Variable(tf.truncated_normal([num_nodes, num_nodes // 2], stddev = 0.1))
    b1 = tf.Variable(tf.truncated_normal([num_nodes // 2], stddev = 0.1))
    
    w2 = tf.Variable(tf.truncated_normal([num_nodes // 2, 2], stddev = 0.1))
    b2 = tf.Variable(tf.truncated_normal([2], stddev = 0.1))

    def model(dataset, steps):
        out_puts, last_state = tf.nn.dynamic_rnn(cell = lstm_cell, dtype = tf.float32, sequence_length = steps, inputs = dataset)
        hidden = last_state[-1]
        hidden = tf.matmul(hidden, w1) + b1
        logits = tf.matmul(hidden, w2) + b2
        return logits
    train_logits = model(tf_train_dataset, tf_train_steps)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf_train_labels, logits = train_logits))
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    test_prediction = tf.nn.softmax(model(tf_test_dataset, tf_test_steps))

num_steps = 300000
summary_frequency = 500
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        offset = (step * batch_size) % (len(lineLabels) - batch_size)
        feed_dict = {tf_train_dataset:lineList[offset:offset + batch_size],
                     tf_train_steps:lineSteps[offset:offset + batch_size],
                     tf_train_labels:lineLabels[offset:offset + batch_size]}
        _, l = session.run([optimizer, loss], feed_dict = feed_dict)
        mean_loss += l
        if step > 0 and step % summary_frequency == 0:
            mean_loss = mean_loss / summary_frequency
            print('The step is:', step)
            print('In the train data, the loss is:%.4f' % mean_loss)
            if mean_loss < 0.03:
                break
            mean_loss = 0
    prediction = session.run(test_prediction)
    data_frame = DataFrame(prediction)
    data_frame.to_csv('./final_result.csv', index = False)
