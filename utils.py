from joblib import Parallel, delayed
from pystoi.stoi import stoi

import shutil
import scipy.io
import librosa
import os
import time  
import numpy as np
import numpy.matlib
import random
import subprocess
from pesq import pesq as pypesq

random.seed(999)

TargetMetric='pesq' # It can be either 'pesq' or 'stoi' for now. Of course, it can be any arbitary metric of interest.
Target_score=np.asarray([1.0]) # Target metric score you want generator to generate. s in e.q. (5) of the paper.

PESQ_path='.'

mask_min=0.05
clipping_constant=10.0  # To prevent clipping of noisy waveform. (i.e., Noisy=(clean+noise)/10)
maxv = np.iinfo(np.int16).max 

def read_pesq_exe(clean_file, enhanced_file, sr):
    try:
        cmd = PESQ_path+'/PESQ_old {} {} +{}'.format(clean_file, enhanced_file, sr)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        out = proc.communicate()
        pesq = float(out[0][-6:-1])
        return (pesq + 0.5) / 5.0
    except Exception as e:
        print("Error:", e, cmd)
        return 0.5
    
def read_pesq(clean_file, enhanced_file, sr):
    try:
        clean_wav, _ = librosa.load(clean_file, sr=16000)     
        enhanced_wav, _ = librosa.load(enhanced_file, sr=16000)
        pesq = pypesq(sr, clean_wav, enhanced_wav, 'wb')
        return pesq
#         return (pesq + 5.0-4.643888473510742) / 5.0
    except Exception as e:
        print(e)
        return 0.5

# Parallel computing for accelerating
def read_batch_PESQ(clean_list, enhanced_list):
    pesq = Parallel(n_jobs=40)(delayed(read_pesq_exe)(clean_list[i], enhanced_list[i], 16000) for i in range(len(enhanced_list)))
    return pesq

def read_batch_PESQ_py(clean_list, enhanced_list):
    pesq = Parallel(n_jobs=5)(delayed(read_pesq)(clean_list[i], enhanced_list[i], 16000) for i in range(len(enhanced_list)))
    return pesq
        
def read_STOI(clean_file, enhanced_file):
    try:
        clean_wav, _ = librosa.load(clean_file, sr=16000)     
        enhanced_wav, _ = librosa.load(enhanced_file, sr=16000)
        stoi_score = stoi(clean_wav, enhanced_wav, 16000, extended=False) 
        return stoi_score
    except Exception as e:
        print("Error:", e, clean_file, enhanced_file)
        return 1.0
    
# Parallel computing for accelerating    
def read_batch_STOI(clean_list, enhanced_list):
    stoi_score = Parallel(n_jobs=30)(delayed(read_STOI)(clean_list[i], enhanced_list[i]) for i in range(len(enhanced_list)))
    return stoi_score

    
def List_concat(score, enhanced_list):
    concat_list=[]
    for i in range(len(score)):
        concat_list.append(str(score[i])+','+enhanced_list[i]) 
    return concat_list
         
def creatdir(directory):    
    if not os.path.exists(directory):
        os.makedirs(directory) 

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
    
    
def Sp_and_phase(signal, Normalization=False):        
    signal_length = signal.shape[0]
    n_fft = 512
    y_pad = librosa.util.fix_length(signal, signal_length + n_fft // 2)
    
    F = librosa.stft(y_pad, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
    
    Lp=np.abs(F)
    phase=np.angle(F)
    if Normalization==True:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257)) # For LSTM
    return NLp, phase, signal_length

def SP_to_wav(mag, phase, signal_length):
    Rec = np.multiply(mag , np.exp(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=256,
                           win_length=512,
                           window=scipy.signal.hamming, length=signal_length)
    return result   

def inv_LP_audio(IRM, noisy_LP, Nphase, signal_length, use_clip=True):
    mask=np.maximum(IRM, mask_min)
    E=np.squeeze(noisy_LP*mask)
    enhanced_wav=SP_to_wav(E.T, Nphase, signal_length)
    if use_clip:
        enhanced_wav=enhanced_wav/clipping_constant
    else:
        enhanced_wav=enhanced_wav/np.max(np.abs(enhanced_wav))
    return enhanced_wav

def Generator_train_data_generator(file_list):
    index=0
    while True:
        noisy_wav = librosa.load(file_list[index], sr=16000)    
        noisy_LP_normalization, _, _= Sp_and_phase(noisy_wav[0]*clipping_constant, Normalization=True)
        noisy_LP, _, _= Sp_and_phase(noisy_wav[0]*clipping_constant, Normalization=False)
         
         
        clean_wav = librosa.load(Train_Clean_path+file_list[index].split('/')[-1], sr=16000)   
        clean_LP, _, _= Sp_and_phase(clean_wav[0]) 

        index += 1
        if index == len(file_list):
            index = 0
            random.shuffle(file_list)
       
        yield [noisy_LP_normalization, noisy_LP.reshape((1,257,noisy_LP.shape[1],1)), clean_LP.reshape((1,257,noisy_LP.shape[1],1)), mask_min*np.ones((1,257,noisy_LP.shape[1],1))], Target_score

def Discriminator_train_data_generator(file_list):
    index=0
    while True:
        score_filepath=file_list[index].split(',')
        noisy_wav = librosa.load(score_filepath[1], sr=16000) 

        if 'dB' in score_filepath[1]:   # noisy or enhanced            
            noisy_LP, _, _ =Sp_and_phase(noisy_wav[0]*clipping_constant)
        else:                          # clean
            noisy_LP, _, _ =Sp_and_phase(noisy_wav[0])

        f=file_list[index].split('/')[-1]
        if '@' in f:
            wave_name=f.split('_')[-1].split('@')[-2]
            clean_wav= librosa.load(Train_Clean_path+'Train_'+wave_name+'.wav', sr=16000) 
            clean_LP, _, _ =Sp_and_phase(clean_wav[0]) 
        else:
            wave_name=f.split('_')[-1]
            clean_wav= librosa.load(Train_Clean_path+'Train_'+wave_name, sr=16000) 
            clean_LP, _, _ =Sp_and_phase(clean_wav[0]) 

        True_score=np.asarray([float(score_filepath[0])])

        index += 1
        if index == len(file_list):
            index = 0

            random.shuffle(file_list)

        yield np.concatenate((noisy_LP.reshape((1,257,noisy_LP.shape[1],1)),clean_LP.reshape((1,257,noisy_LP.shape[1],1))), axis=3), True_score


def Corresponding_clean_list(file_list):
    index=0
    co_clean_list=[]
    while index<len(file_list):
        f=file_list[index].split('/')[-1]
               
        wave_name=f.split('_')[-1]
        clean_name='Train_'+wave_name
            
        co_clean_list.append('1.00,'+Train_Clean_path+clean_name)
        index += 1  
    return co_clean_list