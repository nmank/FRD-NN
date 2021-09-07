import os

'''
a silly little script to make the directories for the dataset
'''

basedir = '/data4/mankovic/FRD-NN/FRD_datasets/mnist/'

for angle in range(0,360,10):
    os.mkdir(basedir + str(angle)+'degrees')
    for num in range(10):
        os.mkdir(basedir + str(angle)+'degrees/'+str(num))