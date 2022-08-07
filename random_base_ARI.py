# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 14:51:56 2022

@author: Takada Satoshi
"""

import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score


z_true = np.loadtxt( "correct_class_color_maze01.txt" )

num_data = len(z_true) #データ数
class_max = max(z_true) #クラスの最大値(クラス数)

#random_class = np.random.randint(0,class_max,len(z_true)) #(low,high,size)
#print(random_class)
#print("ARI:", adjusted_rand_score(random_class, z_true))

ARI_sum = 0
n = 100
for i in range(n):
    random_class = np.random.randint(0,class_max,len(z_true)) #(low,high,size)
    print("ARI:", adjusted_rand_score(random_class, z_true))
    ARI_sum += adjusted_rand_score(random_class, z_true)
    
print("100回の平均ARI:",ARI_sum/n)