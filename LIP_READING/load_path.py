import os
from tqdm import trange
import skvideo
skvideo.setFFmpegPath('C:/Users/Shin/anaconda3/ffmpeg/ffmpeg-2022-04-07-git-607ecc27ed-full_build/bin')
import skvideo.io
import numpy as np
from tensorflow.keras import backend as K

class load_path:
    def total_path():
        mp4_path = "Lip_Reading_in_the_wild/sub_mp4"
        image_path = "Lip_Reading_in_the_wild/lipread_image"
        path = os.getcwd()
        image_path = os.path.join(path, mp4_path)
        mp4_path = os.path.join(path, mp4_path)

        # os.chdir(mp4_path)
        list_mp4 = os.listdir(image_path)
        return mp4_path, list_mp4

    def isWaveFile(name):
        if (name.find("mp4") > -1 or name.find(".MP4") > -1):
            return True
        else:
            return False

    def isValidDir(loc):
        if (os.path.isdir(loc)):
            return True
        else:
            return False

    def load_dir(path, n_class):
        mp4_frames = []
        
        if(load_path.isValidDir(path)):
            list_mp4 = os.listdir(path)
            list_mp4 = [s for s in list_mp4 if (".mp4" in s) or (".MP4" in s)]        
            n_files = len(list_mp4)
            list_mp4 = [os.path.join(path, s) for s in list_mp4]

        elif(os.path.isfile(path)):
            file_wav = [path]

        for j in trange(n_files, desc="loading %s" % n_class):
            loc = os.path.join(path, list_mp4[j])

            try:
                frames = skvideo.io.vreader(loc)
                frames = np.array([frame for frame in frames])
                mp4_frames.append(frames)
            except Exception as e:
                print("\t",e,": file, {}".format(loc))
                pass

        return mp4_frames


    def set_data(frames):
        try:
            data_frames = np.array(frames)
            if (K.image_data_format() == 'channels_first'):
                data_frames = np.rollaxis(data_frames, 3)
            data = data_frames
            length = len(data)
            return data, length
        except:
            pass