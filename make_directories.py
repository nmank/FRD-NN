import os

basedir = './FRD_datasets/mnist/'

for angle in range(0,360,10):
    os.mkdir(basedir + str(angle)+'degrees')
    for num in range(10):
        os.mkdir(basedir + str(angle)+'degrees/'+str(num))