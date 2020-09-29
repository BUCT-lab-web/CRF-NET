import tensorflow as tf
import numpy as np
import time
   
add_data = "unsize_Disjoint_0PolSAR/"  ##########
print ("./Prob_MRF/"+add_data+"train_feature.txt")

learning_rate = 0.0001
training_iters = 20000*1+50  ###########1034
batch_size = 1  ##########1034
display_step = 50

n_input = 109
n_coordinate = 2
n_classes = 11  ##########
dropout = 0.75

test_num = 10

range_row = 750  ##########  
range_col = 1024  ##########

#读取train的feature和labels
Prob_MRF_train = np.loadtxt("./Prob_MRF/"+add_data+"train_feature.txt")
print ('train_feature', Prob_MRF_train.shape)
print("数据类型",type(Prob_MRF_train))           #打印数组数据类型  
print("数组元素数据类型：",Prob_MRF_train.dtype) #打印数组元素数据类型  
Prob_MRF_train = Prob_MRF_train[0:len(Prob_MRF_train)].astype(np.float32)
Prob_MRF_train = np.reshape(Prob_MRF_train, (-1, n_input))
print ('Prob_MRF_train.shape', Prob_MRF_train.shape)
print ('Prob_MRF_train.shape', Prob_MRF_train.shape)

Prob_MRF_train_coordinate = np.loadtxt("./Prob_MRF/"+add_data+"train_row_col.txt")
Prob_MRF_train_coordinate = Prob_MRF_train_coordinate.astype(np.float32)
#np.savetxt("./Prob_MRF/"+add_data+"train_row_col1.txt", Prob_MRF_train_coordinate,"%f")
Prob_MRF_train_coordinate[:, 0] = Prob_MRF_train_coordinate[:, 0]/range_row
Prob_MRF_train_coordinate[:, 1] = Prob_MRF_train_coordinate[:, 1]/range_col
print("数据类型",type(Prob_MRF_train_coordinate))           #打印数组数据类型  
print("数组元素数据类型：",Prob_MRF_train_coordinate.dtype) #打印数组元素数据类型  
# np.savetxt("./Prob_MRF/"+add_data+"train_row_col1.txt", Prob_MRF_train_coordinate,"%f")
print ('Prob_MRF_train_coordinate.shape', Prob_MRF_train_coordinate.shape)
#Prob_MRF_train_coordinate = Prob_MRF_train_coordinate[0:len(Prob_MRF_train_coordinate)-1].astype(np.float32)
#Prob_MRF_train_coordinate = np.reshape(Prob_MRF_train, (-1, n_input))


train_labels = np.loadtxt("./Prob_MRF/"+add_data+"train_label.txt")
train_labels = train_labels.astype(np.float32)
train_labels = np.reshape(train_labels, (-1, 1))#以一列形式显示
print("数据类型",type(train_labels))           #打印数组数据类型  
print("数组元素数据类型：",train_labels.dtype) #打印数组元素数据类型  
#np.savetxt("./Prob_MRF/"+add_data+"train_labels.txt", train_labels,"%f")
print ('train_labels.shape', train_labels.shape)

#转换train的labels为one_hot：保证每个样本中的每个特征只有1位处于状态1,其他都是0
Prob_MRF_train_labels = np.zeros([train_labels.shape[0], n_classes])#677*11
for i in range(Prob_MRF_train_labels.shape[0]):
    Prob_MRF_train_labels[i][int(train_labels[i][0])-1] = 1.
print ('Prob_MRF_train_labels.shape', Prob_MRF_train_labels.shape)


#读取val的feature和labels
Prob_MRF_val= np.loadtxt("./Prob_MRF/"+add_data+"val_feature.txt",  dtype=np.str)
Prob_MRF_val = Prob_MRF_val[0:len(Prob_MRF_val)].astype(np.float32)
Prob_MRF_val = np.reshape(Prob_MRF_val, (-1, n_input))
print ('Prob_MRF_val.shape', Prob_MRF_val.shape)



#val_row_col.txt
Prob_MRF_val_coordinate = np.loadtxt("./Prob_MRF/"+add_data+"val_row_col.txt")
Prob_MRF_val_coordinate = Prob_MRF_val_coordinate.astype(np.float32)
#np.savetxt("./Prob_MRF/"+add_data+"val_row_col1.txt", Prob_MRF_val_coordinate,"%f")
Prob_MRF_val_coordinate[:, 0] = Prob_MRF_val_coordinate[:, 0]/range_row
Prob_MRF_val_coordinate[:, 1] = Prob_MRF_val_coordinate[:, 1]/range_col
print("数据类型",type(Prob_MRF_val_coordinate))           #打印数组数据类型  
print("数组元素数据类型：",Prob_MRF_val_coordinate.dtype) #打印数组元素数据类型  
# np.savetxt("./Prob_MRF/"+add_data+"val_row_col1.txt", Prob_MRF_val_coordinate,"%f")
print ('Prob_MRF_val_coordinate.shape', Prob_MRF_val_coordinate.shape)


#val_label.txt
val_labels = np.loadtxt("./Prob_MRF/"+add_data+"val_label.txt")
val_labels = val_labels.astype(np.float32)
val_labels = np.reshape(val_labels, (-1, 1))
print("数据类型",type(val_labels))           #打印数组数据类型  
print("数组元素数据类型：",val_labels.dtype) #打印数组元素数据类型  
#np.savetxt("./Prob_MRF/"+add_data+"val_labels.txt", val_labels,"%f")
print ('val_labels.shape', val_labels.shape)

#转换val的labels为one_hot
Prob_MRF_val_labels = np.zeros([val_labels.shape[0], n_classes])
for i in range(Prob_MRF_val_labels.shape[0]):
    Prob_MRF_val_labels[i][int(val_labels[i][0])-1] = 1.
print ('Prob_MRF_val_labels.shape', Prob_MRF_val_labels.shape)

print ("======================================")

n_hidden_1 = 100
n_hidden_2 = 100

x = tf.placeholder(tf.float32, [None, n_input])#列是109 行不定
pixel_coordinate = tf.placeholder(tf.float32, [None, n_coordinate])#列是2 行不定
y = tf.placeholder(tf.float32, [None, n_classes])#列是11 行不定
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability) #获取矩阵的值需要调用sess.run( )方法。

weights = {
    #1x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([1, 10, 1, 20])),
    #f_hight, f_width, input_channel, output_channel
    'wd1_1': tf.Variable(tf.random_normal([20*20, 100])),#图片大小400到100 图片大小*通道数
    'wd1_2': tf.Variable(tf.random_normal([n_coordinate, 256])),#先将2维的坐标信息扩展到256维，再将其降到100维
    'wd1_2_1': tf.Variable(tf.random_normal([256, 100])),
    #1024 inputs, 10 outputs
    #LSTM层
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))#100,11
    }

biases = {
    'bc1': tf.Variable(tf.random_normal([20])),
    'bc2': tf.Variable(tf.random_normal([20])),
    'bd1_1': tf.Variable(tf.random_normal([100])),
    'bd1_2': tf.Variable(tf.random_normal([256])),
    'bd1_2_1': tf.Variable(tf.random_normal([100])),
    'out': tf.Variable(tf.random_normal([n_classes]))
    }

#定义卷积层
def conv1d(x, W, b, strides = 1):
    #将2d的卷积操作看成1维的，行的数量等于1
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
#定义max pooling层
def maxpool1d(x, k):
    return tf.nn.max_pool(x, ksize=[1, 1, k, 1], strides=[1, 1, k, 1], padding='VALID')
#tf.nn.max_pool(value, ksize, strides, padding, name=None)
#参数是四个，和卷积很类似：
#第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape

#第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1

#第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]

#第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'



def conv_lstm_net1(x, pixel_coordinate, weights, biases, dropout):
    x_conv = tf.reshape(x, shape=[-1, 1, n_input, 1])#batch, in_height, in_width, in_channels
    
    conv1 = conv1d(x_conv, weights['wc1'], biases['bc1'],)
    print ('conv1',conv1.get_shape())#tensor返回元组 as_list()的操作转换成list
    pool1 = maxpool1d(conv1, k = 5)
    conv1 = pool1[:,:,:20,:] #只取前20个max_pooling后的值（由于向上取整池化结果共有20个值）
    print ('after maxpool',conv1.get_shape())

    fc1 = tf.reshape(conv1, [-1, weights['wd1_1'].get_shape().as_list()[0]])#tf.reshape(conv1,[-1,400])
    print ("*********************************")
    print (weights['wd1_1'].get_shape().as_list())
    print ('fc1',fc1.get_shape())
    fc1 = tf.nn.dropout(fc1, dropout)    
    # tf.matmul(fc1,w)+b
    fc1_1 = tf.nn.relu(tf.add(tf.matmul(fc1, weights['wd1_1']), biases['bd1_1']))
    print ('fc1_1',fc1_1.get_shape())
    
    #pixel coordinate
    fc1_2 = tf.nn.relu(tf.add(tf.matmul(pixel_coordinate, weights['wd1_2']), biases['bd1_2']))
    fc1_2 = tf.nn.relu(tf.add(tf.matmul(fc1_2, weights['wd1_2_1']), biases['bd1_2_1']))#########relu
    print ('fc1_2',fc1_2.get_shape())
    
    fc1 = fc1_1  ### add !!! ###双路
    #fc1 = fc1_1  ### !!! ###单路
    print ('fc1',fc1.get_shape())
    #fc1 = tf.nn.relu(fc1)
    #fc1 = tf.nn.dropout(fc1, dropout)
    
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


pred = conv_lstm_net1(x, pixel_coordinate, weights, biases, keep_prob)
pred1 = tf.nn.softmax(conv_lstm_net1(x, pixel_coordinate, weights, biases, keep_prob))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))#reduce是求均值

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))#equal返回值为true/false
#argmax求最大元素索引值

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))#cast将true和false转化为1和0

init = tf.global_variables_initializer()


for i in range (test_num):
    with tf.Session() as sess:
        sess.run(init)
        out_num = int(training_iters / display_step) - 1
        Testing_Accuarcy = np.zeros([1,2])
        step = 0
        accuracy_test = 0
        rec_loss = np.zeros([out_num,1])
        rec_train_acc = np.zeros([out_num,1])
        rec_test_acc = np.zeros([out_num,1])
        while step * batch_size < training_iters:        
            sess.run(optimizer, feed_dict={x: Prob_MRF_train, pixel_coordinate: Prob_MRF_train_coordinate, \
                                        y: Prob_MRF_train_labels, keep_prob: dropout})
            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: Prob_MRF_train, pixel_coordinate:Prob_MRF_train_coordinate, \
                                                                y: Prob_MRF_train_labels, keep_prob: 1.})
                
                test_acc = sess.run(accuracy, feed_dict={x: Prob_MRF_val, pixel_coordinate: Prob_MRF_val_coordinate, \
                                                        y: Prob_MRF_val_labels, keep_prob: 1.})  
                if(test_acc > accuracy_test):
                            accuracy_test = test_acc
                print ("Epoch" + str(i+1) + ",Iter" + str(step*batch_size) + ", Minibatch Loss=" + "{:.6f}".format(loss) + \
                    ", Training Accuracy=" + "{:.5f}".format(acc) + ", Test Accuracy=" + "{:.5f}".format(test_acc))
                rec = int(step/display_step - 1)
                rec_loss[rec,0] = loss
                rec_train_acc[rec,0] = acc
                rec_test_acc[rec,0] = test_acc
            step += 1
            
        print ("Optimization Finished!!!")
        i_test = str(i)
        np.savetxt("./Prob_MRF/"+add_data+"single_outcome/"+"loss_acc/"+i_test+"rec_loss.txt", rec_loss, '%.6f')
        np.savetxt("./Prob_MRF/"+add_data+"single_outcome/"+"loss_acc/"+i_test+"rec_train_acc.txt", rec_train_acc, '%.6f')
        np.savetxt("./Prob_MRF/"+add_data+"single_outcome/"+"loss_acc/"+i_test+"rec_test_acc.txt", rec_test_acc, '%.6f')
        

        probability = sess.run(pred, feed_dict={x: Prob_MRF_val, pixel_coordinate:Prob_MRF_val_coordinate, \
                                                y: Prob_MRF_val_labels, keep_prob: 1.})
        probability_softmax = sess.run(pred1, feed_dict={x: Prob_MRF_val, pixel_coordinate:Prob_MRF_val_coordinate, \
                                                y: Prob_MRF_val_labels, keep_prob: 1.})
        np.savetxt("./Prob_MRF/"+add_data+"single_outcome/"+"probability/"+i_test+"probability.txt", probability, '%.3f')
        np.savetxt("./Prob_MRF/"+add_data+"single_outcome/"+"probability/"+i_test+"probability_softmax.txt", probability_softmax, '%.3f')
        labels_pred = np.argmax(probability, 1) 
        np.savetxt("./Prob_MRF/"+add_data+"single_outcome/"+"probability/"+i_test+"labels_pred.txt", labels_pred, '%d')

        start_time=time.time()
        accuracy_np = sess.run(accuracy, feed_dict={x: Prob_MRF_val, pixel_coordinate:Prob_MRF_val_coordinate, \
                                                y: Prob_MRF_val_labels, keep_prob: 1.})
        duration = time.time()-start_time 
        duration = duration*60
        print ("Testing Accuarcy:"+"{:.6f}".format(accuracy_np)+ ", Test_time=" + "{:.6f}".format(duration))

        Testing_Accuarcy[0,0] = accuracy_test.astype(np.float32)
        Testing_Accuarcy[0,1] = duration
        np.savetxt("./Prob_MRF/"+add_data+"single_outcome/"+"Testing_Accuarcy/"+i_test+"Testing_Accuarcy.txt", Testing_Accuarcy, '%.6f')