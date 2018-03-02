__author__ = 'jellyzhang'
import tensorflow as tf
import numpy as np
import math
'''
tensorflow实现的mlp类
'''
class Mlp(object):
    def __init__(self,featurecount,layer_number=3,hidden_units=[200,100,300]):
        self.layer_numer=layer_number#层数
        self.hidden_units=hidden_units #每层对应的单元数
        self.input_x=tf.placeholder(shape=[None,featurecount],dtype=tf.float32,name='input_x')
        self.target_y=tf.placeholder(shape=[None,1],dtype=tf.float32,name='target_y')
        self.lr=tf.placeholder(dtype=tf.float32,name='learning_rate')
        self.prob=tf.placeholder(dtype=tf.float32,name='dropout')

        #variables
        #feature_count=self.input_x.get_shape()[1]  #特征数
        weights={
            'h1':tf.Variable(tf.random_normal([featurecount,self.hidden_units[0]])),
            'h2': tf.Variable(tf.random_normal([self.hidden_units[0],self.hidden_units[1]])),
            'h3': tf.Variable(tf.random_normal([self.hidden_units[1],self.hidden_units[2]])),
            'h4': tf.Variable(tf.random_normal([self.hidden_units[2], 1]))
        }
        biass={
            'h1':tf.Variable(tf.random_normal([self.hidden_units[0]])),
            'h2': tf.Variable(tf.random_normal([self.hidden_units[1]])),
            'h3': tf.Variable(tf.random_normal([self.hidden_units[2]])),
            'h4': tf.Variable(tf.random_normal([1]))
        }
        #network
        z1=tf.add(tf.matmul(self.input_x,weights['h1']),biass['h1'])
        a1=tf.nn.relu(z1)
        #a1 = tf.nn.dropout(a1, self.prob)
        z2=tf.add(tf.matmul(a1,weights['h2']),biass['h2'])
        a2=tf.nn.relu(z2)
        #a2 = tf.nn.dropout(a2, self.prob)
        z3 = tf.add(tf.matmul(a2, weights['h3']), biass['h3'])
        a3 = tf.nn.relu(z3)
        #a3 = tf.nn.dropout(a3, self.prob)
        self.logits=tf.add(tf.matmul(a3,weights['h4']),biass['h4'])
        self.loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=self.target_y))
        #梯度裁剪
        tvars=tf.trainable_variables()
        self.opt=tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
        # #计算  <list of variables>相关的梯度
        # grads_and_vars=opt.compute_gradients(self.loss,tvars)
        # #grads_and_vars为tuples（gradient,variable）组成的列表
        # #对梯度进行想要的处理
        # capped_grads_and_vars=[(tf.clip_by_value(grad,-5.,5.),var) for grad,var in grads_and_vars if grad is not None]
        # self.train_op=opt.apply_gradients(capped_grads_and_vars)
        #prediction  用来预测
        self.prediction=tf.sigmoid(self.logits)>0.5
        self.accuracy=tf.reduce_mean(tf.cast(self.prediction,"float"))
    #数据填充
    def train(self,X,Y,epochs=10,batch_size=128,learning_rate=0.005,dropout=0.5):
        #测试集和验证集划分
        split_point=math.ceil(len(X)*0.8)
        test_X,test_Y=X[:split_point],Y[:split_point]
        valid_X,valid_Y=X[split_point:],Y[split_point:]
        #归一化
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        test_X = sc.fit_transform(test_X)
        valid_X = sc.transform(valid_X)

        with tf.Session() as sess:
            #writer = tf.summary.FileWriter("logs/", sess.graph)
            init=tf.global_variables_initializer()
            sess.run(init)
            for e in range(1,epochs+1):
                for batch_index,(x,y) in enumerate(self.get_batchs(test_X,test_Y,batch_size)):
                    feed={self.input_x:x,self.target_y:y,self.lr:learning_rate,self.prob:dropout}
                    loss,_=sess.run([self.loss,self.opt],feed)
                    if batch_index%10==0:
                        val_accuracy=sess.run(self.accuracy,{self.input_x:valid_X,self.target_y:np.transpose(valid_Y[None,:]),self.prob:1.})
                        print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}  - Validation accuracy: {:>6.3f}'
                              .format(e,
                                      epochs,
                                      batch_index,
                                      len(test_X) // batch_size,
                                      loss,val_accuracy))

    def get_batchs(self,X,Y,batch_size=128):
        batches=len(X)//batch_size
        for i in range(batches):
            X_batch=X[i*batch_size:i*batch_size+batch_size]
            Y_batch=np.transpose(Y[i*batch_size:i*batch_size+batch_size][None, :])
            yield X_batch,Y_batch





