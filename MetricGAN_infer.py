# -*- coding: utf-8 -*-
"""
This code (developed with Keras) applies MetricGAN to optimize PESQ or STOI score for Speech Enhancement. 
It can be easily extended to optimize other metrics.

Dependencies:
Python 2.7
keras=2.0.9
librosa=0.5.1


Note:
1) To prevent clipping of noisy waveform (after adding noise to clean speech) 
   when save as .wav, we divide it with a clipping constant 10 (i.e.,Noisy=(clean+noise)/10). 
   Therefore, in this code, there are many operations as *10 and /10 appear in the 
   waveform IO part. This constant should be changed according to the dataset.
2) The PESQ file can only be implemented in Linux environment.


If you find this code useful in your research, please cite:
Citation: 
       [1] S.-W. Fu, C.-F. Liao, Y. Tsao and S.-D. Lin, "MetricGAN: Generative Adversarial Networks based Black-box Metric Scores
           Optimization for Speech Enhancement," in Proc. ICML, 2019.
Contact:
       Szu-Wei Fu
       jasonfu@citi.sinica.edu.tw
       Academia Sinica, Taipei, Taiwan
       
@author: Jason
"""
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json, Model, load_model
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import LSTM, TimeDistributed, Bidirectional, dot, Input, Concatenate, Multiply, Subtract, Maximum
from keras.layers.pooling import GlobalAveragePooling2D
from joblib import Parallel, delayed
from SpectralNormalizationKeras import DenseSN, ConvSN1D, ConvSN2D, ConvSN3D
from pystoi.stoi import stoi
from utils import Sp_and_phase, SP_to_wav, creatdir, inv_LP_audio
import soundfile as sf
from utils import read_batch_STOI, read_batch_PESQ

import shutil
import scipy.io
import librosa
import os
import time  
import numpy as np
import numpy.matlib
import random
import subprocess

random.seed(999)

# input_path='Data/Test/Noisy'
# ref_path='Data/Test/Clean'
# input_path='data/Noisy'
# ref_path='data/Clean'
input_path='data/Noisy'
ref_path=None
output_path='infer_outputs'
reload_model ='current_SE_model.h5'

mask_min=0.05
clipping_constant=10.0  # To prevent clipping of noisy waveform. (i.e., Noisy=(clean+noise)/10)
     
start_time = time.time()
######## Model define start #########
#### Define the structure of Generator (speech enhancement model)  ##### 
print('Generator constructuring...')
de_model = Sequential()

de_model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat', input_shape=(None, 257))) #dropout=0.15, recurrent_dropout=0.15
de_model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat'))

de_model.add(TimeDistributed(Dense(300)))
de_model.add(LeakyReLU())
de_model.add(Dropout(0.05))

de_model.add(TimeDistributed(Dense(257)))
de_model.add(Activation('sigmoid'))

de_model.load_weights(reload_model)

### Define the structure of Discriminator (surrogate loss approximator)  ##### 
print ('Discriminator constructuring...')

_input = Input(shape=(257,None,2))
_inputBN = BatchNormalization(axis=-1)(_input)

C1=ConvSN2D(15, (5,5), padding='valid',  data_format='channels_last') (_inputBN)
C1=LeakyReLU()(C1)

C2=ConvSN2D(25, (7,7), padding='valid',  data_format='channels_last') (C1)
C2=LeakyReLU()(C2)

C3=ConvSN2D(40, (9,9), padding='valid',  data_format='channels_last') (C2)
C3=LeakyReLU()(C3)

C4=ConvSN2D(50, (11,11), padding='valid',  data_format='channels_last') (C3)
C4=LeakyReLU()(C4)

Average_score=GlobalAveragePooling2D(name='Average_score')(C4)  #(batch_size, channels)

D1=DenseSN(50)(Average_score)
D1=LeakyReLU()(D1)

D2=DenseSN(10)(D1)
D2=LeakyReLU()(D2)

Score=DenseSN(1)(D2)

Discriminator = Model(outputs=Score, inputs=_input) 

Discriminator.compile(loss='mse', optimizer='adam')

##### Combine the two networks to become MetricGAN
Clean_reference = Input(shape=(257,None,1))
Noisy_LP        = Input(shape=(257,None,1))
Min_mask        = Input(shape=(257,None,1))

Reshape_de_model_output=Reshape((257, -1, 1))(de_model.output)
Mask=Maximum()([Reshape_de_model_output, Min_mask])

Enhanced = Multiply()([Mask, Noisy_LP]) 
Discriminator_input= Concatenate(axis=-1)([Enhanced, Clean_reference]) # Here the input of Discriminator is (Noisy, Clean) pair, so a clean reference is needed!!

Predicted_score=Discriminator(Discriminator_input) 
 					
MetricGAN= Model(inputs=[de_model.input, Noisy_LP, Clean_reference, Min_mask], outputs=Predicted_score)
MetricGAN.compile(loss='mse', optimizer='adam')

######## Model define end #########

Test_PESQ=[]
Test_STOI=[]
# Test_Predicted_STOI_list=[]
# Train_Predicted_STOI_list=[]
# Previous_Discriminator_training_list=[]

creatdir(os.path.join(output_path, reload_model.split("/")[-1].split(".")[0]))

for (dirpath, dirnames, filenames) in os.walk(input_path):
    for filename in filenames:
        if not filename.endswith(".wav"):
            continue
        enhanced_name = os.path.join(output_path, reload_model.split("/")[-1].split(".")[0], filename)
        noisy_wav, _ = librosa.load(os.path.join(dirpath, filename), sr=16000, mono=True, duration=10.0)
        noisy_LP_normalization, Nphase, signal_length=Sp_and_phase(noisy_wav*clipping_constant, Normalization=True)
        noisy_LP, _, _= Sp_and_phase(noisy_wav*clipping_constant)

        IRM=de_model.predict(noisy_LP_normalization)
        enhanced_wav = inv_LP_audio(IRM, noisy_LP, Nphase, signal_length, use_clip=False)
        sf.write(enhanced_name, enhanced_wav, 16000)
        if ref_path is not None:
            ref_filename = os.path.join(dirpath, filename).replace(input_path, ref_path)

            # Calculate True STOI    
            test_STOI=read_batch_STOI([ref_filename], [enhanced_name])     
            Test_STOI.append(np.mean(test_STOI))
            # Calculate True PESQ    
            test_PESQ=read_batch_PESQ([ref_filename], [enhanced_name])         
            Test_PESQ.append(np.mean(test_PESQ)*5.-0.5)
            print(enhanced_name, np.mean(test_STOI), np.mean(test_PESQ)*5.-0.5)

end_time = time.time()
print('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))
print("STOI:", np.mean(np.array(Test_STOI)), "PESQ:", np.mean(np.array(Test_PESQ)))
