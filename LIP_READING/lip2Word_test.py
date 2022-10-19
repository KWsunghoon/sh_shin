
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
import os

def resize_frame(image, size, offset) :
    '''
    Resize the image from the center point
    Args :
        - image : numpy ndarray representing the image
        - size : size of the image
        - offset : shift of the middle point
    Returns :
        - a numpy ndarrray representing the resized image
    '''
    resized_image = cv2.resize(image, (64, 64))
    return resized_image

def get_label_from_path(path, label_dict) :
    '''
    Find the label from the path of the .mp4
    Args :
        - path : path to the .mp4
        - label_dict : a dict to match a word to a label
    Returns :
        - label (int)
    '''
    return label_dict[path.split('\\')[6]]

def create_dict_word_list(path) :
    '''
    Create a dict used to transfrom labels from str to int
    Args :
        - path : Path to the word list
    Return :
        - Python dictionnary {Word : Label}
    '''
    count = 0
    my_dict = dict()
    with open(path+'word_list.txt', 'r') as f:
        for line in f:
            my_dict.update({line[:-1] : count})
            count += 1
    return my_dict

def capture_process_frames(path, size) :
    '''
    Captures and processes all the frames from a video
    Args :
        - path : path to the .mp4 file
        - size : size of the image
    Returns 
        - a vector representing the video (all frames, concatenated along a third dimension (time))
    '''
    vidObj = cv2.VideoCapture(path) 
    count = 0
    success = 1
    size_frame = 256 # size of the original frame from the video
    number_frames = 29 # all videos are 29 frames
    all_frames = np.zeros((size*number_frames, size))
    while success: 
        success, image = vidObj.read()
        if success :
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = resize_frame(image, size_frame-180, offset=35)
            image = cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            all_frames[count*size:size*(count+1), :] = image
            count += 1
    return all_frames
count = 0

# Create empty matrix
image = np.zeros((29*112, 112)).astype(np.float32)

# Iterate over .mp4 files in every sub directory (train, val, test)
pathlist = Path('C:/Users/Shin/anaconda3/Lip_Reading_in_the_wild/lipread_test').glob('*/*.mp4')
label_dict = create_dict_word_list('C:/Users/Shin/anaconda3/Lip_Reading_in_the_wild/Lip2Word-master/data/')
for path in tqdm(pathlist):
    image = capture_process_frames(str(path), 112) 
    cv2.imwrite('C:/Users/Shin/anaconda3/Lip_Reading_in_the_wild/Lip2Word-master/model/'+'{}/{}_{}_{}.jpg'.format('test', 
                                                              get_label_from_path(str(path), label_dict), 
                                                              str(path).split('\\')[6],   
                                                              count), image)
    count += 1