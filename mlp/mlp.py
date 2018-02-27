__author__ = 'jellyzhang'
import tensorflow as tf
import numpy as np
'''
tensorflow实现的mlp类
'''
class Mlp(object):
    def __init__(self,layer_numer=3,hidden_units=[200,100,300]):
        self.layer_numer-layer_numer #层数
        self.hidden_units=hidden_units #每层对应的单元数


