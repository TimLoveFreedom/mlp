__author__ = 'jellyzhang'
import tensorflow as tf
import numpy as np
import math
'''
tensorflow实现的mlp类
'''
class Mlp(object):
    def __init__(self,layer_number=3,hidden_units=[200,100,300]):
        self.layer_numer=layer_number#层数
        self.hidden_units=hidden_units #每层对应的单元数
        self.input_x=tf.placeholder(shape=[None,None],dtype=tf.int32,name='input_x')
        self.target_y=tf.placeholder(shape=[None,None],dtype=tf.int32,name='target_y')
        self.lr=tf.placeholder(dtype=tf.float32,name='learning_rate')
        self.prob=tf.placeholder(dtype=tf.float32,name='dropout')

        #variables
        feature_count=input_x.get_shape()[1]  #特征数
        weights={
            'h1':tf.Variable(tf.random_norm([feature_count,self.hidden_units[0]])),
            'h2': tf.Variable(tf.random_norm([self.hidden_units[0],self.hidden_units[1]])),
            'h3': tf.Variable(tf.random_norm([self.hidden_units[1],self.hidden_units[2]]))
        }
        biass={
            'h1':tf.Variable(tf.random_norm([self.hidden_units[0]])),
            'h2': tf.Variable(tf.random_norm([self.hidden_units[1]])),
            'h3': tf.Variable(tf.random_norm([self.hidden_units[2]]))
        }
        #network
        z1=tf.add(tf.matmul(input_x,weights['h1']),biass['h1'])
        z1=tf.nn.dropout(z1,self.prob)
        a1=tf.nn.relu(z1)
        z2=tf.add(tf.matmul(a1,weights['h2']),biass['h2'])
        z2=tf.nn.dropout(z2,self.prob)
        a2=tf.nn.relu(z2)
        self.logits=tf.add(tf.matmul(a2,weights['h3']),biass['h3'])
        self.loss=tf.nn.sigmoid_cross_entropy_with_logits(self.logits,self.target_y)
        #梯度裁剪
        tvars=tf.trainable_variables()
        opt=tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        #计算  <list of variables>相关的梯度
        grads_and_vars=opt.compute_gradients(self.loss,tvars)
        #grads_and_vars为tuples（gradient,variable）组成的列表
        #对梯度进行想要的处理
        capped_grads_and_vars=[(tf.clip_by_value(grad,-5.,5.),var) for grad,var in grads_and_vars if grad is not None]
        self.train_op=opt.apply_gradients(capped_grads_and_vars)
        #prediction  用来预测
        self.prediction=tf.sigmoid(self.logits)
    #数据填充
    def train(self,X,Y,epochs=10,batch_size=128,learning_rate=0.001,dropout=0.5):
        #测试集和验证集划分
        split_point=math.ceil(len(X)*0.8)
        test_X,test_Y=X[:split_point],Y[:split_point]
        valid_X,valid_Y=X[split_point:],Y[split_point:]
        with tf.Session() as sess:
            #writer = tf.summary.FileWriter("logs/", sess.graph)
            init=tf.global_variables_initializer()
            sess.run(init)
            for e in range(1,epochs+1):
                for batch_index,(x,y) in enumerate(self.get_batchs(test_X,test_Y,batch_size)):
                    feed={self.input_x:x,self.target_y:y,self.lr:learning_rate,self.prob:dropout}
                    loss,_=sess.run([self.loss,self.train_op],feed)
                    if batch_index%10==0:
                        val_loss=sess.run(self.loss,{self.input_x:valid_X,self.target_y:valid_Y,self.prob:1.})
                        print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                              .format(e,
                                      epochs,
                                      batch_index,
                                      len(test_X) // batch_size,
                                      loss,val_loss))

    def get_batchs(self,X,Y,batch_size=128):
        batches=len(X)//batch_size
        for i in range(batch_size):
            X_batch=X[i*batch_size,i*batch_size+batch_size]
            Y_batch=Y[i*batch_size,i*batch_size+batch_size]
            yield X_batch,Y_batch





