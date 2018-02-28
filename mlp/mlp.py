__author__ = 'jellyzhang'
import tensorflow as tf
import numpy as np
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
        a1=tf.nn.relu(z1)
        z2=tf.add(tf.matmul(a1,weights['h2']),biass['h2'])
        a2=tf.nn.relu(z2)
        self.logits=tf.add(tf.matmul(a2,weights['h3']),biass['h3'])
        self.loss=tf.nn.sigmoid_cross_entropy_with_logits(self.logits,self.target_y)
        #梯度裁剪
        tvars=tf.trainable_variables()
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=self.lr)


    #数据填充
    def train(self,X,Y,learning_rate=0.001,dropout=0.5):
        loss=self.get_loss()






