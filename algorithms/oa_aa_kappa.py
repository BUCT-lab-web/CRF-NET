import argparse
import auxil.mydata as mydata
import auxil.mymetrics as mymetrics
import gc
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.layers import Activation, BatchNormalization, Conv3D, Dense, Flatten, MaxPooling3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import to_categorical as keras_to_categorical
import numpy as np
import sys

add_data1 = "india_16class_1034_rand1/" 
add_data2 = "unsize_Disjoint_0Indian/"
add_data3 = "PaviaU_13class_4306/" 
add_data4 = "unsize_Disjoint_0PaviaU/" 
add_data5 = "PolSAR/" 
add_data6 = "unsize_Disjoint_0PolSAR/" 

add_data01 = "india" 
add_data02 = "Dis_india"
add_data03 = "PaviaU" 
add_data04 = "Dis_PaviaU" 
add_data05 = "PolSAR" 
add_data06 = "Dis_PolSAR" 

indian_labels0 = np.loadtxt("indian_val_label0.txt")
indian_labels0 = indian_labels0.astype(np.float32)
indian_labels0.reshape(-1)
pavia_labels0 = np.loadtxt("pavia_val_label0.txt")
pavia_labels0 = pavia_labels0.astype(np.float32)
pavia_labels0.reshape(-1)
sar_labels0 = np.loadtxt("sar_val_label0.txt")
sar_labels0 = sar_labels0.astype(np.float32)
sar_labels0.reshape(-1)

indian_labels = np.loadtxt("indian_val_label.txt")
indian_labels = indian_labels.astype(np.float32)
indian_labels.reshape(-1)
pavia_labels = np.loadtxt("pavia_val_label.txt")
pavia_labels = pavia_labels.astype(np.float32)
pavia_labels.reshape(-1)
sar_labels = np.loadtxt("sar_val_label.txt")
sar_labels = sar_labels.astype(np.float32)
sar_labels.reshape(-1)

indian_labels0 = indian_labels0[indian_labels0!=0] - 1
pavia_labels0 = pavia_labels0[pavia_labels0!=0] - 1
sar_labels0 = sar_labels0[sar_labels0!=0] - 1
indian_labels = indian_labels[indian_labels!=0] - 1
pavia_labels = pavia_labels[pavia_labels!=0] - 1
sar_labels = sar_labels[sar_labels!=0] - 1


all_mean0 = np.zeros([6,19])
all_mean1 = np.zeros([6,12])
all_mean2 = np.zeros([6,14])
oak = np.zeros([6,3])

num1_class = 16
num2_class = 9
num3_class = 11


for j in range(6):
    if j == 0: 
        stats = np.ones((10, num1_class+3)) * -1000.0 # OA, AA, K, Aclass
        for i in range (10):
            pres = np.loadtxt("./Prob_MRF/"+add_data1+"dual_outcome/probability/"+str(i)+"labels_pred.txt")
            pres = pres.astype(np.float32)
            pres.reshape(-1)
            stats[i,:] = mymetrics.reports(pres, indian_labels0)[2]
        oa_std = np.std(stats[:,0],ddof=1)
        aa_std = np.std(stats[:,1],ddof=1)
        kappa_std = np.std(stats[:,2],ddof=1)
        all_mean0[j,:] = np.mean(stats,axis=0)
        oak[j,0]=oa_std
        oak[j,1]=aa_std
        oak[j,2]=kappa_std

    if j == 1: 
        stats = np.ones((10, num1_class+3)) * -1000.0 # OA, AA, K, Aclass
        for i in range (10):
            pres = np.loadtxt("./Prob_MRF/"+add_data2+"dual_outcome/probability/"+str(i)+"labels_pred.txt")
            pres = pres.astype(np.float32)
            pres.reshape(-1)
            stats[i,:] = mymetrics.reports(pres, indian_labels)[2]
        oa_std = np.std(stats[:,0],ddof=1)
        aa_std = np.std(stats[:,1],ddof=1)
        kappa_std = np.std(stats[:,2],ddof=1)
        all_mean0[j,:] = np.mean(stats,axis=0)
        oak[j,0]=oa_std
        oak[j,1]=aa_std
        oak[j,2]=kappa_std

    if j == 2: 
        stats = np.ones((10, num2_class+3)) * -1000.0 # OA, AA, K, Aclass
        for i in range (10):
            pres = np.loadtxt("./Prob_MRF/"+add_data3+"dual_outcome/probability/"+str(i)+"labels_pred.txt")
            pres = pres.astype(np.float32)
            pres.reshape(-1)
            stats[i,:] = mymetrics.reports(pres, pavia_labels0)[2]
        oa_std = np.std(stats[:,0],ddof=1)
        aa_std = np.std(stats[:,1],ddof=1)
        kappa_std = np.std(stats[:,2],ddof=1)
        all_mean1[j,:] = np.mean(stats,axis=0)
        oak[j,0]=oa_std
        oak[j,1]=aa_std
        oak[j,2]=kappa_std

    if j == 3: 
        stats = np.ones((10, num2_class+3)) * -1000.0 # OA, AA, K, Aclass
        for i in range (10):
            pres = np.loadtxt("./Prob_MRF/"+add_data4+"dual_outcome/probability/"+str(i)+"labels_pred.txt")
            pres = pres.astype(np.float32)
            pres.reshape(-1)
            stats[i,:] = mymetrics.reports(pres, pavia_labels)[2]
        oa_std = np.std(stats[:,0],ddof=1)
        aa_std = np.std(stats[:,1],ddof=1)
        kappa_std = np.std(stats[:,2],ddof=1)
        all_mean1[j,:] = np.mean(stats,axis=0)
        oak[j,0]=oa_std
        oak[j,1]=aa_std
        oak[j,2]=kappa_std

    if j == 4: 
        stats = np.ones((10, num3_class+3)) * -1000.0 # OA, AA, K, Aclass
        for i in range (10):
            pres = np.loadtxt("./Prob_MRF/"+add_data5+"dual_outcome/probability/"+str(i)+"labels_pred.txt")
            pres = pres.astype(np.float32)
            pres.reshape(-1)
            stats[i,:] = mymetrics.reports(pres, sar_labels0)[2]
        oa_std = np.std(stats[:,0],ddof=1)
        aa_std = np.std(stats[:,1],ddof=1)
        kappa_std = np.std(stats[:,2],ddof=1)
        all_mean2[j,:] = np.mean(stats,axis=0)
        oak[j,0]=oa_std
        oak[j,1]=aa_std
        oak[j,2]=kappa_std

    if j == 5: 
        stats = np.ones((10, num3_class+3)) * -1000.0 # OA, AA, K, Aclass
        for i in range (10):
            pres = np.loadtxt("./Prob_MRF/"+add_data6+"dual_outcome/probability/"+str(i)+"labels_pred.txt")
            pres = pres.astype(np.float32)
            pres.reshape(-1)
            stats[i,:] = mymetrics.reports(pres, sar_labels)[2]
        oa_std = np.std(stats[:,0],ddof=1)
        aa_std = np.std(stats[:,1],ddof=1)
        kappa_std = np.std(stats[:,2],ddof=1)
        all_mean2[j,:] = np.mean(stats,axis=0)
        oak[j,0]=oa_std
        oak[j,1]=aa_std
        oak[j,2]=kappa_std

print("                                                                         ")
print("Data"+"\t"+"\t"+"OA"+"\t"+"\t"+"AA"+"\t"+"\t"+"Kappa")
for k in range (6):
    if k ==0:
        print(add_data01+"\t"+"\t"+"{:.2f}".format(all_mean0[k,0])+"+"+"{:.2f}".format(oak[k,0])+"\t"+"{:.2f}".format(all_mean0[k,1])+"+"+"{:.2f}".format(oak[k,1])+"\t"+"{:.2f}".format(all_mean0[k,2])+"+"+"{:.2f}".format(oak[k,2])+"\t"+"{:.2f}".format(all_mean0[k,3])+"\t"+"{:.2f}".format(all_mean0[k,4])+"\t"+"{:.2f}".format(all_mean0[k,5])+"\t"+"{:.2f}".format(all_mean0[k,6])+"\t"+"{:.2f}".format(all_mean0[k,7])+"\t"+"{:.2f}".format(all_mean0[k,8])+"\t"+"{:.2f}".format(all_mean0[k,9])+"\t"+"{:.2f}".format(all_mean0[k,10])+"\t"+"{:.2f}".format(all_mean0[k,11])+"\t"+"{:.2f}".format(all_mean0[k,12])+"\t"+"{:.2f}".format(all_mean0[k,13])+"\t"+"{:.2f}".format(all_mean0[k,14])+"\t"+"{:.2f}".format(all_mean0[k,15])+"\t"+"{:.2f}".format(all_mean0[k,16])+"\t"+"{:.2f}".format(all_mean0[k,17])+"\t"+"{:.2f}".format(all_mean0[k,18]))
    if k ==1:
        print(add_data02+"\t"+"{:.2f}".format(all_mean0[k,0])+"+"+"{:.2f}".format(oak[k,0])+"\t"+"{:.2f}".format(all_mean0[k,1])+"+"+"{:.2f}".format(oak[k,1])+"\t"+"{:.2f}".format(all_mean0[k,2])+"+"+"{:.2f}".format(oak[k,2])+"\t"+"{:.2f}".format(all_mean0[k,3])+"\t"+"{:.2f}".format(all_mean0[k,4])+"\t"+"{:.2f}".format(all_mean0[k,5])+"\t"+"{:.2f}".format(all_mean0[k,6])+"\t"+"{:.2f}".format(all_mean0[k,7])+"\t"+"{:.2f}".format(all_mean0[k,8])+"\t"+"{:.2f}".format(all_mean0[k,9])+"\t"+"{:.2f}".format(all_mean0[k,10])+"\t"+"{:.2f}".format(all_mean0[k,11])+"\t"+"{:.2f}".format(all_mean0[k,12])+"\t"+"{:.2f}".format(all_mean0[k,13])+"\t"+"{:.2f}".format(all_mean0[k,14])+"\t"+"{:.2f}".format(all_mean0[k,15])+"\t"+"{:.2f}".format(all_mean0[k,16])+"\t"+"{:.2f}".format(all_mean0[k,17])+"\t"+"{:.2f}".format(all_mean0[k,18]))
    if k ==2:
        print(add_data03+"\t"+"\t"+"{:.2f}".format(all_mean1[k,0])+"+"+"{:.2f}".format(oak[k,0])+"\t"+"{:.2f}".format(all_mean1[k,1])+"+"+"{:.2f}".format(oak[k,1])+"\t"+"{:.2f}".format(all_mean1[k,2])+"+"+"{:.2f}".format(oak[k,2])+"\t"+"{:.2f}".format(all_mean1[k,3])+"\t"+"{:.2f}".format(all_mean1[k,4])+"\t"+"{:.2f}".format(all_mean1[k,5])+"\t"+"{:.2f}".format(all_mean1[k,6])+"\t"+"{:.2f}".format(all_mean1[k,7])+"\t"+"{:.2f}".format(all_mean1[k,8])+"\t"+"{:.2f}".format(all_mean1[k,9])+"\t"+"{:.2f}".format(all_mean1[k,10])+"\t"+"{:.2f}".format(all_mean1[k,11]))
    if k ==3:
        print(add_data04+"\t"+"{:.2f}".format(all_mean1[k,0])+"+"+"{:.2f}".format(oak[k,0])+"\t"+"{:.2f}".format(all_mean1[k,1])+"+"+"{:.2f}".format(oak[k,1])+"\t"+"{:.2f}".format(all_mean1[k,2])+"+"+"{:.2f}".format(oak[k,2])+"\t"+"{:.2f}".format(all_mean1[k,3])+"\t"+"{:.2f}".format(all_mean1[k,4])+"\t"+"{:.2f}".format(all_mean1[k,5])+"\t"+"{:.2f}".format(all_mean1[k,6])+"\t"+"{:.2f}".format(all_mean1[k,7])+"\t"+"{:.2f}".format(all_mean1[k,8])+"\t"+"{:.2f}".format(all_mean1[k,9])+"\t"+"{:.2f}".format(all_mean1[k,10])+"\t"+"{:.2f}".format(all_mean1[k,11]))
    if k ==4:
        print(add_data05+"\t"+"\t"+"{:.2f}".format(all_mean2[k,0])+"+"+"{:.2f}".format(oak[k,0])+"\t"+"{:.2f}".format(all_mean2[k,1])+"+"+"{:.2f}".format(oak[k,1])+"\t"+"{:.2f}".format(all_mean2[k,2])+"+"+"{:.2f}".format(oak[k,2])+"\t"+"{:.2f}".format(all_mean2[k,3])+"\t"+"{:.2f}".format(all_mean2[k,4])+"\t"+"{:.2f}".format(all_mean2[k,5])+"\t"+"{:.2f}".format(all_mean2[k,6])+"\t"+"{:.2f}".format(all_mean2[k,7])+"\t"+"{:.2f}".format(all_mean2[k,8])+"\t"+"{:.2f}".format(all_mean2[k,9])+"\t"+"{:.2f}".format(all_mean2[k,10])+"\t"+"{:.2f}".format(all_mean2[k,11])+"\t"+"{:.2f}".format(all_mean2[k,12])+"\t"+"{:.2f}".format(all_mean2[k,13]))
    if k ==5:
        print(add_data06+"\t"+"{:.2f}".format(all_mean2[k,0])+"+"+"{:.2f}".format(oak[k,0])+"\t"+"{:.2f}".format(all_mean2[k,1])+"+"+"{:.2f}".format(oak[k,1])+"\t"+"{:.2f}".format(all_mean2[k,2])+"+"+"{:.2f}".format(oak[k,2])+"\t"+"{:.2f}".format(all_mean2[k,3])+"\t"+"{:.2f}".format(all_mean2[k,4])+"\t"+"{:.2f}".format(all_mean2[k,5])+"\t"+"{:.2f}".format(all_mean2[k,6])+"\t"+"{:.2f}".format(all_mean2[k,7])+"\t"+"{:.2f}".format(all_mean2[k,8])+"\t"+"{:.2f}".format(all_mean2[k,9])+"\t"+"{:.2f}".format(all_mean2[k,10])+"\t"+"{:.2f}".format(all_mean2[k,11])+"\t"+"{:.2f}".format(all_mean2[k,12])+"\t"+"{:.2f}".format(all_mean2[k,13]))





# print("indian:",list(stats1[-1]))
# print("pavia:",list(stats2[-1]))
# print("sar:",list(stats3[-1]))


























