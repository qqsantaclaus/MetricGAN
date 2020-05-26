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
import keras
from keras.models import Sequential, model_from_json, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape, Lambda
from keras.layers import BatchNormalization
from keras.layers import ELU, PReLU, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import LSTM, TimeDistributed, Bidirectional, dot, Input, Concatenate, Multiply, Maximum
from keras.layers import GlobalAveragePooling2D
from joblib import Parallel, delayed
from SpectralNormalizationKeras import DenseSN, ConvSN1D, ConvSN2D, ConvSN3D
from pystoi.stoi import stoi
from utils import Sp_and_phase, SP_to_wav, creatdir
from pesq import pesq as pypesq
from data_reader import DataGenerator
from speech_embedding.emb_data_generator import query_joint_yield, CLEAN_DATA_RANGE, CLEAN_TEST_DATA_RANGE
from speech_embedding.read_Audio_RIRs import read_Audio_RIRs, get_Audio_RIR_classes, read_noise
import tensorflow as tf

import shutil
import scipy.io
import librosa
import os
import time  
import numpy as np
import numpy.matlib
import random
import subprocess
from tqdm import tqdm

random.seed(999)

tf.test.is_gpu_available()

TargetMetric='pesq' # It can be either 'pesq' or 'stoi' for now. Of course, it can be any arbitary metric of interest.
Target_score=np.asarray([1.0]) # Target metric score you want generator to generate. s in e.q. (5) of the paper.

output_path='train_outputs'
# PESQ_path='.'

GAN_epoch=200
mask_min=0.05
num_of_sampling=1600
num_of_disc_sample=400
# num_of_valid_sample=1000
clipping_constant=10.0  # To prevent clipping of noisy waveform. (i.e., Noisy=(clean+noise)/10)

maxv = np.iinfo(np.int16).max 

def read_pesq(clean, enhanced, sr):
    try:
        pesq = pypesq(sr, clean, enhanced, 'wb')
        return (pesq + 5.0-4.643888473510742) / 5.0
    except Exception as e:
        return 1.0
#     return (pesq+0.5)/5.0

# Parallel computing for accelerating
def read_batch_PESQ(clean_list, enhanced_list):
    pesq = Parallel(n_jobs=10)(delayed(read_pesq)(clean_list[i], enhanced_list[i], 16000) for i in range(len(enhanced_list)))
    return pesq
        
def read_STOI(clean_wav, enhanced_wav):
    try:
        stoi_score = stoi(clean_wav, enhanced_wav, 16000, extended=False) 
        return stoi_score
    except Exception as e:
        return 1.0
    
# Parallel computing for accelerating    
def read_batch_STOI(clean_list, enhanced_list):
    stoi_score = Parallel(n_jobs=10)(delayed(read_STOI)(clean_list[i], enhanced_list[i]) for i in range(len(enhanced_list)))
    return stoi_score
    
def List_concat(score, enhanced_list):
    concat_list=[]
    for i in range(len(score)):
        concat_list.append(str(score[i])+','+enhanced_list[i]) 
    return concat_list

def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path
     
def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory. 

def zip_discriminator_data(score_list, noisy_list, clean_list):
    score_list = score_list + ([1.0] * len(score_list))
    noisy_list = noisy_list + clean_list
    clean_list = clean_list + clean_list
    print(score_list)
    comb = list(zip(score_list, noisy_list, clean_list))
    random.shuffle(comb)
    print(len(comb))
    return comb

def Discriminator_train_data_generator(data_list):
    while True:
        for index in range(len(data_list)):
            true_score = np.asarray([float(data_list[index][0])])
            audio_wav = data_list[index][1]
            clean_wav = data_list[index][2]
            noisy_LP, _, _ =Sp_and_phase(audio_wav)       
            clean_LP, _, _ =Sp_and_phase(clean_wav)
#             print(true_score.shape)
            yield np.concatenate((noisy_LP.reshape((1,257,noisy_LP.shape[1],1)),clean_LP.reshape((1,257,noisy_LP.shape[1],1))), axis=3), true_score
        random.shuffle(data_list)

# def Corresponding_clean_list(file_list):
#     index=0
#     co_clean_list=[]
#     while index<len(file_list):
#         f=file_list[index].split('/')[-1]
               
#         wave_name=f.split('_')[-1]
#         clean_name='Train_'+wave_name
            
#         co_clean_list.append('1.00,'+Train_Clean_path+clean_name)
#         index += 1  
#     return co_clean_list

# def LP_audio(d_noisy):
#     noisy_LP_normalization, Nphase, signal_length=Sp_and_phase(d_noisy*clipping_constant, Normalization=True)
#     noisy_LP, _, _= Sp_and_phase(d_noisy*clipping_constant)

#     return noisy_LP_normalization, noisy_LP, Nphase, signal_length

def inv_LP_audio(IRM, noisy_LP_normalization, noisy_LP, Nphase, signal_length):
    mask=np.maximum(IRM, mask_min)
    E=np.squeeze(noisy_LP*mask)
    enhanced_wav=SP_to_wav(E.T,Nphase, signal_length)
    enhanced_wav=enhanced_wav/np.max(abs(enhanced_wav))
    return enhanced_wav
    
#########################  Training data #######################
SR = 16000
IN_MEMORY = 0.0
DIRECTORY = "/trainman-mount/trainman-storage-dc5e03f8-a08d-49bb-b3a9-4bae92eb4e92"
REVERB_DIRECTORY = "/trainman-mount/trainman-storage-420a420f-b7a2-4445-abca-0081fc7108ca/Audio-RIRs"
NOISE_DIRECTORY = "/trainman-mount/trainman-storage-420a420f-b7a2-4445-abca-0081fc7108ca/subnoises"
VAL_NOISE_DIRECTORY = "/trainman-mount/trainman-storage-420a420f-b7a2-4445-abca-0081fc7108ca/subnoises"
data_range = CLEAN_DATA_RANGE
test_data_range = CLEAN_TEST_DATA_RANGE

NO_CLASSES = 200
SEQ_LEN = 32000
BATCH_SIZE = 8
# LEARNING_RATE = config['LEARNING_RATE']
# path = config['path']
# dropout = config['dropout'] if "dropout" in config else 0.0
inject_noise = True  
use_real_noise = True
augment_speech = False
augment_reverb = False
augment_noise = False
reload_model = "current_SE_model.h5"
extra_subsets = False
    
#%% Speech audio files
train_filenames, train_data_holder = query_joint_yield(
                                         gender=data_range["gender"], 
                                         num=data_range["num"], 
                                         script=data_range["script"],
                                         device=data_range["device"], 
                                         scene=data_range["scene"], 
                                         directory=DIRECTORY, 
                                         exam_ignored=True, 
                                         randomized=True,
                                         sample_rate=SR, 
                                         in_memory=IN_MEMORY)

test_filenames, test_data_holder = query_joint_yield(
                                         gender=test_data_range["gender"], 
                                         num=test_data_range["num"], 
                                         script=test_data_range["script"],
                                         device=test_data_range["device"], 
                                         scene=test_data_range["scene"], 
                                         directory=DIRECTORY, 
                                         exam_ignored=True, 
                                         randomized=True,
                                         sample_rate=SR,
                                         in_memory=IN_MEMORY)
print(test_filenames)

if extra_subsets:
    unseen_speaker_test_filenames, _ = query_joint_yield(
                                         gender=test_data_range["gender"], 
                                         num=test_data_range["num"], 
                                         script=data_range["script"],
                                         device=test_data_range["device"], 
                                         scene=test_data_range["scene"], 
                                         directory=DIRECTORY, 
                                         exam_ignored=True, 
                                         randomized=True,
                                         sample_rate=SR,
                                         in_memory=IN_MEMORY)

    unseen_script_test_filenames, _ = query_joint_yield(
                                         gender=data_range["gender"], 
                                         num=data_range["num"], 
                                         script=test_data_range["script"],
                                         device=test_data_range["device"], 
                                         scene=test_data_range["scene"], 
                                         directory=DIRECTORY, 
                                         exam_ignored=True, 
                                         randomized=True,
                                         sample_rate=SR,
                                         in_memory=IN_MEMORY)

#%% Reverb audio files
# 1-250 Train
# 251-271 Test
# 233 is missing
reverb_train_filenames, reverb_train_data_holder=read_Audio_RIRs(sr=SR, 
                                                                 subset="train", 
                                                                 cutoff=NO_CLASSES, 
                                                                 root=REVERB_DIRECTORY)
reverb_test_filenames, reverb_test_data_holder=read_Audio_RIRs(sr=SR, 
                                                               subset="test", 
                                                               cutoff=NO_CLASSES, 
                                                               root=REVERB_DIRECTORY)

print(len(reverb_train_filenames), len(reverb_test_filenames))

#%% Target classes
class_dict, classes, class_back_dict = get_Audio_RIR_classes(REVERB_DIRECTORY, 271)

### Prepare Noise
if use_real_noise:
    noise_filenames, _ = read_noise(sr=SR, root=NOISE_DIRECTORY, preload=False)
    val_noise_filenames, _ = read_noise(sr=SR, root=VAL_NOISE_DIRECTORY, preload=False)
#     print(val_noise_filenames)
else:
    noise_filenames = None
    val_noise_filenames = None
    
train_set_generator = DataGenerator(train_filenames, reverb_train_filenames, noise_filenames=noise_filenames,
                                speech_data_holder=None, 
                                reverb_data_holder=None,
                                noise_data_holder=None,
                                sample_rate=SR, 
                                seq_len=SEQ_LEN, 
                                num_classes=NO_CLASSES,
                                shuffle=True, batch_size=BATCH_SIZE,
                                in_memory=1.0, 
                                augment_speech=augment_speech, inject_noise=inject_noise, augment_reverb=augment_reverb)

val_set_generator = DataGenerator(test_filenames, reverb_train_filenames, noise_filenames=val_noise_filenames,
                                speech_data_holder=None, 
                                reverb_data_holder=None,
                                noise_data_holder=None,
                                sample_rate=SR, 
                                seq_len=SEQ_LEN, 
                                num_classes=NO_CLASSES,
                                shuffle=True, batch_size=BATCH_SIZE,
                                in_memory=1.0, 
                                augment_speech=augment_speech, inject_noise=inject_noise, augment_reverb=augment_reverb)

if extra_subsets:
    val_unseen_speaker_set_generator = DataGenerator(unseen_speaker_test_filenames, reverb_train_filenames, 
                                    noise_filenames=val_noise_filenames,
                                    speech_data_holder=None, 
                                    reverb_data_holder=None,
                                    noise_data_holder=None,
                                    sample_rate=SR, 
                                    seq_len=SEQ_LEN, 
                                    num_classes=NO_CLASSES,
                                    shuffle=True, batch_size=BATCH_SIZE,
                                    in_memory=1.0, 
                                    augment_speech=augment_speech, inject_noise=inject_noise, augment_reverb=augment_reverb)

    val_unseen_script_set_generator = DataGenerator(unseen_script_test_filenames, reverb_train_filenames, 
                                    noise_filenames=val_noise_filenames,
                                    speech_data_holder=None, 
                                    reverb_data_holder=None,
                                    noise_data_holder=None,
                                    sample_rate=SR, 
                                    seq_len=SEQ_LEN, 
                                    num_classes=NO_CLASSES,
                                    shuffle=True, batch_size=BATCH_SIZE,
                                    in_memory=1.0, 
                                    augment_speech=augment_speech, inject_noise=inject_noise, augment_reverb=augment_reverb)

test_set_generator = DataGenerator(test_filenames, reverb_test_filenames, noise_filenames=val_noise_filenames,
                                    speech_data_holder=None, 
                                    reverb_data_holder=None,
                                    noise_data_holder=None,
                                    sample_rate=SR, 
                                    seq_len=SEQ_LEN, 
                                    num_classes=NO_CLASSES,
                                    shuffle=True, batch_size=BATCH_SIZE,
                                    in_memory=1.0, 
                                    augment_speech=augment_speech, inject_noise=inject_noise, augment_reverb=augment_reverb)

g1 = train_set_generator.__iter__()
d_train_noisy_iter = train_set_generator.__iter_raw__()
d_val_noisy_iter = val_set_generator.__iter_raw__()

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

print(de_model.summary())
#### Define the structure of Discriminator (surrogate loss approximator)  ##### 
print('Discriminator constructuring...')

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

Discriminator.trainable = True 
Discriminator.compile(loss='mse', optimizer='adam')


#### Combine the two networks to become MetricGAN
Discriminator.trainable = False 
  
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

if reload_model is not None:
    de_model.load_weights(reload_model)

######## Model define end #########
Test_STOI = []
Test_PESQ = []

Previous_Discriminator_training_list=[]
shutil.rmtree(output_path)

for gan_epoch in np.arange(1, GAN_epoch+1):
    
    # Prepare directories
    creatdir(output_path+"/epoch"+str(gan_epoch))
    creatdir(output_path+"/epoch"+str(gan_epoch)+"/"+"Test_epoch"+str(gan_epoch))
    creatdir(output_path+'/For_discriminator_training')
    creatdir(output_path+'/temp')
    
#     random sample some training data  
#     random.shuffle(Generator_Train_Noisy_paths)
#     g1 = Generator_train_data_generator(Generator_Train_Noisy_paths[0:num_of_sampling])
             
    print('Epoch', gan_epoch, ': Generator training (with discriminator fixed)...')
    if gan_epoch>=2:              
        Generator_hist = MetricGAN.fit_generator(g1, steps_per_epoch=num_of_sampling, 
                                epochs=1,
                                verbose=1,
                                max_queue_size=1, 
                                workers=1,
                                )

    # Evaluate the performance of generator in a validation set.
    print('Evaluate G by validation data ...')   
    Test_enhanced_list=[]
    Test_clean_list=[]
    utterance=0
    for i in tqdm(range(10)):
        d_noisy, d_clean, noisy_LP_normalization, noisy_LP, Nphase, signal_length, clean_name = next(d_val_noisy_iter) 
        IRM=de_model.predict(noisy_LP_normalization)
        enhanced_wav = inv_LP_audio(IRM, noisy_LP_normalization, noisy_LP, Nphase, signal_length)
        
        if utterance<3: # Only seperatly save the firt 20 utterance for listening comparision 
            enhanced_name=output_path+"/epoch"+str(gan_epoch)+"/"+"Test_epoch"+str(gan_epoch)+"/"+ str(utterance) +"@"+str(gan_epoch)+".wav"
        else:           # others will be overrided to save hard disk memory.
            enhanced_name=output_path+"/temp"+"/"+str(utterance)+"@"+str(gan_epoch)+".wav"
        librosa.output.write_wav(enhanced_name, enhanced_wav, 16000)
        utterance+=1
        Test_enhanced_list.append(enhanced_wav)  
        Test_clean_list.append(d_clean/np.max(np.abs(d_clean)))
              
    # Calculate True STOI    
    test_STOI=read_batch_STOI(Test_clean_list, Test_enhanced_list)     
    print(np.mean(test_STOI))    
    Test_STOI.append(np.mean(test_STOI))
    
    # Calculate True PESQ    
    test_PESQ=read_batch_PESQ(Test_clean_list, Test_enhanced_list)         
    print(np.mean(test_PESQ)*5.-0.5)
    Test_PESQ.append(np.mean(test_PESQ)*5.-0.5)
    
#     # Plot learning curves
#     plt.figure(1)
#     plt.plot(range(1,gan_epoch+1),Test_STOI,'b',label='ValidPESQ')
#     plt.xlim([1,gan_epoch])
#     plt.xlabel('GAN_epoch')
#     plt.ylabel('STOI')
#     plt.grid(True)
#     plt.show()
#     plt.savefig('Test_STOI.png', dpi=150)
    
#     plt.figure(2)
#     plt.plot(range(1,gan_epoch+1),Test_PESQ,'r',label='ValidPESQ')
#     plt.xlim([1,gan_epoch])
#     plt.xlabel('GAN_epoch')
#     plt.ylabel('PESQ')
#     plt.grid(True)
#     plt.show()
#     plt.savefig('Test_PESQ.png', dpi=150)
    
    # save the current SE model
    de_model.save('current_SE_model.h5')     

    print('Sample training data for discriminator training...')
    Test_enhanced_list=[]
    Test_clean_list=[]
    
    for i in tqdm(range(num_of_disc_sample)):
        d_noisy, d_clean, noisy_LP_normalization, noisy_LP, Nphase, signal_length = next(d_train_noisy_iter) 
        IRM=de_model.predict(noisy_LP_normalization)
        enhanced_wav = inv_LP_audio(IRM, noisy_LP_normalization, noisy_LP, Nphase, signal_length)
        Test_enhanced_list.append(enhanced_wav)
        Test_clean_list.append(d_clean/np.max(np.abs(d_clean)))
    
    score_list = []
    per_share = 40
    for i in tqdm(range(num_of_sampling // per_share)):
        s_ind = per_share * i
        e_ind = per_share * (i+1)
        if TargetMetric=='stoi':
            # Calculate True STOI score   
            score_list.extend(read_batch_STOI(Test_clean_list[s_ind:e_ind], Test_enhanced_list[s_ind:e_ind]))
        elif TargetMetric=='pesq':
            # Calculate True PESQ score
            score_list.extend(read_batch_PESQ(Test_clean_list[s_ind:e_ind], Test_enhanced_list[s_ind:e_ind]))
#     print(score_list)

    print('Discriminator training...')                            
    ## Training for current list    
    Current_Discriminator_training_list = zip_discriminator_data(score_list, Test_enhanced_list, Test_clean_list)
    d_current          = Discriminator_train_data_generator(Current_Discriminator_training_list)  
    Discriminator_hist = Discriminator.fit_generator(d_current, steps_per_epoch=len(Current_Discriminator_training_list),
                                epochs=1, verbose=1,
                                max_queue_size=1, 
                                workers=1,
                                )

    ## Training for current list + Previous list (like replay buffer in RL, optional)                           
    random.shuffle(Previous_Discriminator_training_list)
    Total_Discriminator_training_list=Previous_Discriminator_training_list[:len(Previous_Discriminator_training_list)//10]\
                    +Current_Discriminator_training_list # Discriminator_Train_list is the list used for pretraining.
    
    random.shuffle(Total_Discriminator_training_list)
    d_current_past     = Discriminator_train_data_generator(Total_Discriminator_training_list)
    Discriminator_hist = Discriminator.fit_generator(d_current_past, steps_per_epoch=len(Total_Discriminator_training_list), 
                                epochs=1, verbose=1,
                                max_queue_size=1, 
                                workers=1,
                                )
    
    # Update the history list
    Previous_Discriminator_training_list=Previous_Discriminator_training_list+Current_Discriminator_training_list 
    
    ## Training current list again (optional)   
    Discriminator_hist = Discriminator.fit_generator(d_current, steps_per_epoch=len(Current_Discriminator_training_list), 
                                epochs=1, verbose=1,
                                max_queue_size=1, 
                                workers=1,
                                )
                                
    shutil.rmtree(output_path+'/temp') # to save harddisk memory

end_time = time.time()
print('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))