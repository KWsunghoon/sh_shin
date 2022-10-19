#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
from multiprocessing import Queue
import sys
import os
import tqdm
import pickle
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa
from matplotlib.animation import FuncAnimation
from datetime import date, datetime
from tqdm import trange
from ast import literal_eval

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers, utils

print(('TensorFlow version: {0}').format(tf.__version__))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

#%%
# TF status
# if(tf.executing_eagerly()):
#     print(('\nEager execution is enabled (running operations immediatly)\n'))
#     print(('You can turn eager-execution off by running: \n{0} \n{1}').format('\tfrom tensorflow.python.framework.ops\n\timport disagle_eager_execution', '\tdiable_eager_execution()\n'))
# else:
#     print('\nYou are not running eager execution. TensorFlow version >= 2.0.0' \
#           'has eager execution enabled by default.')
#     print(('\nTurn on eager execution by running: \n\n{0}\n\nOr upgrade '\
#            'your tensorflow version by running:\n\n{1}').format(
#            'tf.compat.v1.enable_eager_execution()',
#            '!pip install --upgrade tensorflow\n' \
#            '!pip install --upgrade tensorflow-gpu'))
print("Eager execution: %s"%str(tf.executing_eagerly()))

# print(('\nIs your GPU available for use?\n{0}').format(
#     'Yes, your GPU is available: True' if tf.config.list_physical_devices('GPU') != [] else 'No, your GPU is NOT available: False'))
# print(('\n\tYour devices that are available:\n{0}').format(
#     '\n'.join('{}'.format(item) for item in tf.config.list_physical_devices()[1::2])))


# parameters
# command-line arguments setting
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-L', '--List-devices', action='store_true',
    help='show list of audio devices and exit')

args, remaining = parser.parse_known_args()
if args.List_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])

parser.add_argument(
    '-sd', '--source_device', type=int_or_str,
    help='input device (numeric ID or substring)')

parser.add_argument(
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')

# parameters for realtime plotting
parser.add_argument(
    '-i', '--interval', type=float, default=1,
    help='minimum time between plot updates (default: %(default)ms)')

parser.add_argument(
    '-sr', '--samplerate', type=float, default=48000,
    help='sampling rate of the input')

parser.add_argument(
    '-n', '--downsample', type=int, default=1, metavar='N',
    help='display every Nth sample (default: %(default))')


# if you need to refresh the saved dataset pickle
parser.add_argument(
    "-c", "--clear_dataset", type=str, default='False', metavar="ex) True",
    help = "reset the dataset pickle")


# arguments for preprocessing
parser.add_argument(
    "-s", "--source_dir", type=str, metavar="ex) wav",
    help = "input source (default: %(default))", default='wav')

parser.add_argument(
    "channels", type=int, default=2, nargs='*', metavar='ex) 2',
    help='input channels for plotting and reading(default: %(default)second')

parser.add_argument(
    "-o", "--wav_offset", type=float, default=0.5,
    help='reading point offset(default: %(default)second)')

parser.add_argument(
    "--size_fft", type=int, default=1024,
    help="length of fft point(default:%(default)points")

parser.add_argument(
    "--bin_from", type=int, default=1,
    help="starting filter-bin from MFCC(default:%(default)")

parser.add_argument(
    "--bin_to", type=int, default=29,
    help="last filter-bin from MFCC(default:%(default)")

parser.add_argument(
    "--n_frame", type=int, default=20,
    help="number of frames for one input block(1/samplingrate*size_fft = one frame), default:%(default)")

parser.add_argument(
    "--n_overlap", type=int, default=2,
    help="MFCC overapping ratio(=int(size_fft/ n_overalp)), default:%(default)")

parser.add_argument(
    '-d', "--duration", type=float, default=60,
    help='duration for observing (default: %(default)second)')

parser.add_argument(
    '-bs', "--blocksize", type=int,
    help='block size for sample slicing',
    default=2205)

parser.add_argument(
    "-q", "--len_seq", type=int, default=2400,
    help = "Length of a input sample slice")


# arguments for training NN

parser.add_argument(
    "-m", "--mode", type=str, default="training",
    help="select operation mode, ex) training or mic")
parser.add_argument(
    "-lr", "--learning_rate", type=float, default=1e-3,
    help="initial learning-rate, default:%(default)")

parser.add_argument(
    "-b", "--n_batch", type=int, default=16,
    help="size of batch normalization, default:%(default)")

parser.add_argument(
    "-e", "--n_epoch", type=int, default=5,
    help="numver of epoch for training, default:%(default)")


args = parser.parse_args(remaining)

# parameters for preprocessing
device = args.source_device
window = args.window
source_dir = args.source_dir
offset_reading = args.wav_offset
sr = args.samplerate
bin_to = args.bin_to
bin_from = args.bin_from
wav_offset = args.wav_offset
refresh_dataset = True if args.clear_dataset == "True" else False
n_bin = args.bin_to - args.bin_from
n_fft = args.size_fft
n_frame = args.n_frame
duration = args.duration
n_overlap = int(n_fft/2)
n_step = n_overlap * (n_frame-1)
n_stride = n_overlap
if any(c < 1 for c in range(args.channels+1)[1:]):
    parser.error('argument CHANNEL: must be >= 1')
channels = [c for c in range(args.channels+1)[1:]]  # Channel numbers start with 1
mapping = [c - 1 for c in channels]
mode = args.mode
lr = args.learning_rate
epochs = args.n_epoch
n_batch = args.n_batch
len_seq = args.len_seq





#%%
# setting working directory
def isWaveFile(name):
    return True if(name.find(".wav") > -1 or name.find(".WAV") > -1) else False

def isValidDir(loc):
    return True if(os.path.isdir(loc)) else False

dir_code = os.getcwd()
# if os.name == 'nt':
#     dir_code = os.getcwd()
# else :
#     dir_code = "/home/dclabxavier2/tf"

curr_date = str(date.today())
curr_time = str(time.strftime("%H%M%S", time.localtime()))

try:
    os.mkdir(os.path.join(dir_code, curr_date))
except OSError as error:
    print('\t',error,": directory exists already")
    pass

dir_base = os.path.join(dir_code, curr_date)

try:
    os.mkdir(os.path.join(dir_base, "pkl"))
    os.mkdir(os.path.join(dir_base, "dataset"))
except OSError as error:
    print(error)
    pass

dir_pkl = os.path.join(dir_base, "pkl")
dir_wav = os.path.join(dir_code, args.source_dir)
# pkl_wav = "waveform.pkl"
pkl_dataset = "waveform.pkl"
pkl_dataset = "dataset.pkl"
# file_wav = dir_wav if(isValidDir(dir_wav)) else exit()


if(os.path.isfile(os.path.join(dir_pkl, "dataset.pkl")) and (os.path.getsize(os.path.join(dir_pkl, "dataset.pkl")) == 0)):
        os.remove(os.path.join(dir_pkl, "dataset.pkl"))


if os.path.exists(dir_wav):
    print("\n\tdir_wav is valid")
else:
    sys.exit("\n\twave source is not exist. check the dir_wav")

if os.path.exists(dir_pkl):
    print("\ndir_pkl is valid")
else:
    print("\n\twave SOURCE is not exist, making it...")
try:
    os.mkdir(dir_pkl)
except Exception as e:
    print(e)
    print("\ndir_pkl is valid")



if(isWaveFile(args.source_dir)):
    dir_wav = dir_base
    label_class = ["single_file_test",]
 
    n_class = 1

else:
    os.chdir(dir_wav)
    label_class = os.listdir()
    label_class = [s for s in label_class if(isValidDir(s) and not isWaveFile(s))]

# n_class = len(label_class) + 1

if(label_class != []):
    n_class = len(label_class)
else:
    print("check your data file structure")
    exit(1)



#%%
# Helper functions
def clear_dataset(loc_dir=dir_pkl, loc_file=pkl_dataset):
    pkl_dataset = os.path.join(loc_dir, loc_file)
    # if(os.path.isdir(loc_dir)):
    #     os.rmdir(loc_dir)
    if(os.path.isfile(pkl_dataset)):
        os.remove(pkl_dataset)
    else:
        pass

# ### Helper-function for loading and building dataset
def load_wav_dir(s_class, dir_wav, duration=duration):
    global sr
    
    start_time = time.time()

    
    # listup files from given dir_wav
    if(isValidDir(dir_wav)):
        list_wav = os.listdir(dir_wav)
        list_wav = [s for s in list_wav if (".wav" in s) or (".WAV" in s)]
        n_files = len(list_wav)
        list_wav = [os.path.join(dir_wav, s) for s in list_wav]
        sr_list = [librosa.get_samplerate(s) for s in list_wav]
    elif(os.path.isfile(dir_wav)):
        file_wav = [dir_wav]
    
    # loading wav files and putting waveforms into waves
    n_samples = 0
    wfs = []
    _duration = duration / n_files
    
    # _duration = duration
    for j in trange(n_files, desc="loading class %s..." % s_class):
        # _offset = wav_offset + _duration*j
        loc = os.path.join(dir_wav, list_wav[j])
        try:
            wf, sr = librosa.load(
                    loc,
                    sr=sr,
                    mono=False,
                    offset=wav_offset,
                    duration=_duration,
                    res_type='kaiser_best')
            # If it's stereo signal
            if(np.array(wf.shape)[0] == 2):
                wf = (wf[0,:] + wf[1,:]) / 2
                wf = wf/wf.max() + 1e-16
                pass
            else:
                wf = wf/wf.max() + 1e-16
                pass
            wf = list(wf)
            if(n_samples < duration*sr*1.2):
                n_samples += len(wf)
                wfs = wfs + wf
                wfs = wfs[0:int(duration*sr)]    
                pass
            else:
                wfs = wfs[0:int(duration*sr)]    
                break
        except Exception as e:
            print("\t",e,": file, {}".format(loc))
            
            # list_wav.remove(list_wav[j])
            pass
        
        
        

    print("total duration: %2.3f seconds"%(len(wfs)/sr))

    end_time = time.time()
    print("took {0:.1f} seconds...\n".format(end_time-start_time))

    # return wfs_class, {"class_type":class_type, "len_total":len_total, "n_files":n_files, "power_wfs":np.power(wfs_in_dir, 2).mean()}
    return wfs


def wav_to_mfcc(wf, sr, label=-1):
    print("\n--- mfccing from waves...\n")
    start_time = time.time()
    n_wf = len(wf)
    
    n_sample = len(wf)
    n_symbol = n_sample // n_step
    mfcc = np.zeros((n_symbol, n_bin, n_frame))
    for j in trange(n_symbol, desc="making mfcc for "+label_class[label]):
        _from = (j*n_stride)
        _to = _from + n_step
        if( (_to-_from) < (n_step-2) ):
            break
        # print("--- mean: {0}, median: {1}".format(
        #     np.mean(wf[i][1][_from:_to]), np.median(wf[i][1][_from:_to])))
        _mfcc = librosa.feature.mfcc(
            y=np.asarray(wf[_from:_to]),
            sr=sr,
            n_fft=n_fft,
            n_mfcc=40,
            htk=True)
        mfcc[j,:,:] = _mfcc[bin_from:bin_to]
        # _mfcc[j,bin_from:bin_to,:] = np.clip(
        #     _mfcc[j,bin_from:bin_to,:], a_min=-50, a_max=100)
        min_block = _mfcc[bin_from:bin_to].min()
        max_block = _mfcc[bin_from:bin_to].max()
        # mfcc[j,:,:] = (mfcc[j,:,:] - min_block) / (max_block - min_block) + 1e-16
        mfcc[j,:,:] = (mfcc[j,:,:] - min_block) + 1e-16
        
        # _mfcc[j,:,:] = ( _zero_min * (255/_zero_min.max()) ).astype(np.uint8)
    
    end_time = time.time()
    print("\ntook {0:.1f} seconds...".format(end_time-start_time))
    
    if label == -1:
        return mfcc
    else:
        return [label, mfcc]



def serialize(mfcc):
    print("\n--- Serializing dataset...\n")
    start_time = time.time()
    list_n_symbols = [mfcc[s][1].shape[0] for s in range(len(mfcc))] 
    total_n_symbols = sum(list_n_symbols)
    dataset = np.zeros((total_n_symbols, n_bin*n_frame+1))
    
    idx = 0
    for i in range(n_class):
        j = list_n_symbols[i]
        for k in range(j):
            dataset[idx,:] = list(mfcc[i][1][k].reshape(n_bin*n_frame)) + [i]
            idx += 1
        k = 0
        
    end_time = time.time()
    print("\ntook {0:.1f} seconds...".format(end_time-start_time))
    
    return dataset, dataset.shape
    
        
        
def mfcc_to_image(mfcc, label_class):
    print("\n--- Dumping the dataset into images...\n")
    start_time = time.time()
    idx = 0
    for i in trange(len(label_class)):
        for j in trange(len(mfcc[i])):
            for k in range(len(mfcc[i][j][1])):
                _zero_min = mfcc[i][j][1][k] - mfcc[i][j][1][k].min()
                _mfcc = _zero_min * (255/_zero_min.max())
                img = Image.fromarray( _mfcc.astype(np.uint8) )
                loc = os.path.join(dir_img, \
                    label_class[i], mfcc[i][j][0][:-4]+str(k)+".png")
                img.save(loc)
                idx += 1
    end_time = time.time()
    print("\ntook {0:.1f} seconds...".format(end_time-start_time))

    return idx



def one_hot(i, n_class=n_class):
    return np.squeeze(np.eye(n_class)[i])

def load_dataset(dir, name):
    loc = os.path.join(dir, name)
    if(os.path.isfile(loc)):
        try:
            with open(loc, "rb") as file:
                dataset = pickle.load(file)
                print("----------- Dataset loaded successfully.\n")
        except:
            sys.exit("----------- Failed to load saved pickle: "+name+"\n")
    else:
        sys.exit("----------- Failed to load saved pickle: "+name+"\n")

    return dataset



def save_dataset(data, dir, name):
    loc = os.path.join(dir, name)
    try:
        with open(loc, 'wb') as file:
            pickle.dump(data, file)
            print("\n----------- {0} stored successfully: {1}\n".format(name, loc))
    
    except Exception as e:
        sys.exit("\n----------- E:{0}, Failed from saving dataset: {1}\n".format(str(e), loc))


def build_dataset(mode="slices", train=.7, test=.3):
    label_file = []
    n_files = []
    mfcc = []

    for i in range(n_class):
        loc = os.path.join(dir_wav, label_class[i])
        _files = os.listdir(loc)
        _files = [s for s in _files if not ".ini" in s]
        label_file.append(_files)
        n_files.append([label_class[i], len(_files)])

    # label_file.append("None")


    # if img_enabled:
    #     if os.path.exists(dir_img):
    #         print("\ndir_img is valid")
    #     else:
    #         os.mkdir(dir_img)
    #         print("\ndir_img created")

    #     # directories for images
    #     for i in range(n_class):
    #         loc = os.path.join(dir_img, label_class[i])
    #         if os.path.exists(loc):
    #             print("\n{} is valid".format(loc))
    #         else:
    #             try:
    #                 os.mkdir(loc)
    #                 print("\n{} dir_img created".format(loc))
    #             except:
    #                 sys.exit("\nfailed to create directory:", loc)

    try:
        wfs = load_pkl(dir_pkl, "waveform.pkl")
    except Exception:
        print("\t", Exception)
        print("\n\tloading waves...\n")
        wfs = []
        n_samples = []
        for i, _class in zip(range(n_class), label_class):
            loc = os.path.join(dir_wav, _class)
            if (len(os.listdir(loc))>0):
                _wfs = load_wav_dir(_class, loc, duration=duration)
                # _wfs = np.asarray(_wfs)
                wfs.append(_wfs)
            n_samples.append(len(_wfs))
        dist = list(np.asarray(n_samples)/sum(n_samples))
        
        save_pkl(wfs, dir_pkl, "waveform.pkl")
        
        print("\tInput distributions:")
        for i in range(len(n_files)):
            print("\t\t{0}, {1}".format(label_class[i], dist[i]))
    
    try:
        dataset = load_pkl(dir_pkl, "dataset.pkl")
    except Exception:
        print("\t", Exception)
        for i in range(n_class):
            _mfcc = wav_to_mfcc(wfs[i], sr, i)
            mfcc.append(_mfcc)
        # mfcc = np.asarray(mfcc)
        dataset, _dim = serialize(mfcc)
        print("\tTotal {0} rows have been made.".format(_dim[0]))
        # save_dataset(dataset, dir_pkl, "dataset"+str(_dim[0])+".pkl")
        save_dataset(dataset, dir_pkl, "dataset.pkl")

    if(mode=="solid"):
        return dataset, _dim
    elif(mode=="slices"):
        # dataset = dataset.astype(np.float16)
        np.random.shuffle(dataset)
        # np.random.shuffle(dataset)
        n_samples = dataset.shape[0]
        s, l = dataset[:,0:-1], dataset[:,-1]
        d = np.array([(d!=label_ambient_noise)*1 for d in l])
        l = np.array([one_hot(int(s)) for s in l])
        
        
        n_train = int(n_samples * train)
        # n_test = int(n_samples * test)
        x_train = s[0:n_train]
        x_test = s[n_train+1:n_samples]
        y_cls_train = l[0:n_train]
        y_cls_test = l[n_train+1:n_samples]
        y_det_train = d[0:n_train]
        y_det_test = d[n_train+1:n_samples]
        
        return x_train, x_test, y_cls_train, y_cls_test, y_det_train, y_det_test
    elif(mode=="x_y"):
        np.random.shuffle(dataset)
        # np.random.shuffle(dataset)
        n_samples = dataset.shape[0]
        s, l = dataset[:,0:-1], dataset[:,-1]
        d = np.array([(d!=label_ambient_noise)*1 for d in l])
        l = np.array([one_hot(int(s)) for s in l])
        
        return s, [l, d]
        

    

def load_pkl(loc_dir, loc_file):
    start_time = time.time()
    if(os.path.isfile(loc_dir)):
        loc = loc_dir
    else:
        loc = os.path.join(loc_dir, loc_file)
    try:
        with open(loc, "rb") as file:
            pkl = pickle.load(file)
            end_time = time.time() - start_time
            print("\t{0} loaded successfully: in {1} seconds...".format(loc, end_time))
            return pkl
    except OSError as error:
        print("\t", error, "\n:Failed to load: "+ loc + "\n")
        raise error
        return None
    

def save_pkl(data, loc, filename):
    loc = os.path.join(loc, filename)
    try:
        with open(loc, 'wb') as file:
            pickle.dump(data, file)
            print("\n\t{0} stored successfully: {1}\n".format(filename, loc))
    
    except Exception as e:
        sys.exit("\terror:{0}, Failed saving the pickle file: {1}\n".format(str(e), loc))
        os.remove(loc)


def load_saved_model(loc=dir_pkl, name="model.h5"):
    file = os.path.join(loc, name)
    try:
        model = models.load_model(file)
        print("\tModel %s has been loaded."%loc)
        pass
    except Exception as e:
        print("\t Loading model %s canceled. Check the model file first.", loc)
        pass
    
    return model
    

label_file = []
n_files = []
label_ambient_noise = 0

for i in range(n_class):
    if(isValidDir(dir_wav)):
        loc = os.path.join(dir_wav, label_class[i])
        if(loc.find("noise") > 0):
            label_ambient_noise = i
        _files = os.listdir(loc)
        _files = [s for s in _files if(isWaveFile(s))]
        label_file.append(_files)
        n_files.append([label_class[i], len(_files)])
    else:
        _files = os.path.join(dir_wav, label_class[i]) + ".wav"


if(args.clear_dataset == "True"):
    clear_dataset()
    pass
else:
    pass


5.97
14.96

modelname = "_model.h5"

if mode=="training" :


    x_train, x_test, y_cls_train, y_cls_test, y_det_train, y_det_test = build_dataset("slices")
    # np.random.shuffle(pkl_loaded)
    # waveforms = [s for (s, l) in pkl_loaded]
    # labels = [l for (s, l) in pkl_loaded]
    # # waveforms = {"input":waveforms}
    # waveforms = np.asarray(waveforms)
    # waveforms = np.reshape(waveforms, (waveforms.shape[0], 1, len_seq,))
    # # waveforms = np.expand_dims(waveforms, axis=0)

    # labels_class = [one_hot(l) for (s, l) in pkl_loaded]
    # labels_detec = []
    # for l in labels:
    #     if l == label_ambient_noise:
    #         labels_detec.append([0])
    #         pass
    #     else:
    #         labels_detec.append([1])
    # # labels = {"pred_classify":labels_class,
    # #           "pred_detec":labels_detec}
    # labels_class = np.asarray(labels_class)
    # labels_detec = np.asarray(labels_detec)


    # layer_input = layers.Input(shape=(None, len_seq))

    # layer_conv1d_1 = layers.Conv1D(254, 256, strides=1, padding='same', activation='relu', name="conv1d_1")(layer_input)
    # # layer_maxpooling1d_1 = layers.MaxPooling1D(pool_size=4)(layer_conv1d_prep)
    # layer_conv1d_2 = layers.Conv1D(100, 256, strides=1, padding='same', name="conv1d_2")(layer_conv1d_1)

    layer_input = layers.Input(shape=(None, x_train.shape[1]))

    # layer_reshape_2d_1 = layers.Reshape((10, 10, 1,))(layer_conv1d_2)

    layer_reshape_2d_1 = layers.Reshape((n_bin, n_frame, 1))(layer_input)

    # layer_flatten_0 = layers.Flatten()(layer_reshape_2d_1)

    # layer_dense_1 = layers.Dense(64)(layer_flatten_0)

    layer_conv2d_1 = layers.Conv2D(filters=32,
                                kernel_size=5,
                                strides=(1,1),
                                padding='same',
                                activation='relu',
                                name="conv2d_1")(layer_reshape_2d_1)
    layer_maxpooling_1 = layers.MaxPool2D(pool_size=(2, 2),
                                        strides=None,
                                        padding='valid',
                                        name="maxpool_1")(layer_conv2d_1)
    layer_conv2d_2 = layers.Conv2D(filters=64,
                                kernel_size=5,
                                strides=(1,1),
                                padding='same',
                                activation='relu',
                                name="conv2d_2")(layer_maxpooling_1)
    layer_maxpooling_2 = layers.MaxPool2D(pool_size=(2, 2),
                                        strides=None,
                                        padding='valid',
                                        name="maxpool_2")(layer_conv2d_2)

    layer_flatten_1 = layers.Flatten()(layer_maxpooling_2)
    # layer_flatten_2 = layers.Flatten()(layer_reshape_2d_1)

    # layer_dense_concat = layers.Concatenate(axis=-1)([layer_flatten_1, layer_flatten_2])

    layer_dense_2 = layers.Dense(128)(layer_flatten_1)

    layer_drop_1 = layers.Dropout(rate=.2)(layer_dense_2)

    dense_classify = layers.Dense(n_class, name="pred_classify", activation='softmax')(layer_drop_1) 
    dense_detec = layers.Dense(1., name="pred_detection")(layer_drop_1)


    model = tf.keras.Model(
        inputs = [layer_input],
        outputs = [dense_classify, dense_detec])

    list_callback = [
        # keras.callbacks.EarlyStopping(
        #     monitor='val_pred_classify_accuracy',
        #     patience=30
        # ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(dir_pkl, "model.h5"),
            monitor='val_pred_classify_accuracy',
            save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_pred_detection_loss',
            factor=.1,
            patience=30,
            mode='auto',
            verbose=1
        )
    ]


    # opt = keras.optimizers.RMSprop(learning_rate=0.001, clipvalue=0.5)
    opt = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        loss=['categorical_crossentropy', 'binary_crossentropy'],
        loss_weights=[1., .3],
        optimizer=opt,
        metrics=['accuracy']
    )

    model.summary()
    # utils.plot_model(model, to_file="conv2d_with_detec.png", show_shapes=True)
    
            
    hist = model.fit(
    x = x_train,
    y = [y_cls_train, y_det_train],
    epochs=epochs,
    validation_split=.2,
    batch_size=128,
    callbacks=list_callback)
    
    model_json = model.to_json()
    with open(os.path.join(dir_pkl, "hybrid_a_0_2_3.json"), "w") as file:
        file.write(model_json)
    model.save(os.path.join(dir_pkl, modelname))

    print(model.evaluate(x_test, [y_cls_test, y_det_test]))

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_pred_classify_loss'], 'r', label='val cls loss')
    loss_ax.plot(hist.history['val_pred_detection_loss'], 'm', label='val detec loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='lower right')

    acc_ax.plot(hist.history['pred_classify_accuracy'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_pred_classify_accuracy'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left')
    plt.show()
    
    pass
elif mode=="stream" :
    # IPC helper class
    class ADQ_server():
            
        def __init__(self, path='./adq_buffer.pipe'):
            self.path = path
            self.AI0 = None
            self.AI1 = None
            self.AI2 = None
            self.AI3 = None
            
        def run(self):
            try:
                print('-> StartReading')
        
                aiData = open(self.path).read()
                adq = literal_eval(aiData)
                self.AI0 = adq['Data'][0]['AI0']['Voltage']
                self.AI1 = adq['Data'][0]['AI1']['Voltage']
                self.AI2 = adq['Data'][0]['AI2']['Voltage']
                self.AI3 = adq['Data'][0]['AI3']['Voltage']
        
                print('-> EndReadingProc')

            except Exception as e:
                print(e, 'Incorrect ADQ data')

        def get_singlechannel(self, channel=0):
            self.run()
            if(channel == 0):
                return self.AI0
            elif(channel == 1):
                return self.AI1
            elif(channel == 2):
                return self.AI2
            elif(channel == 3):
                return self.AI3
            else:
                print("\tError: check the channel==??)")
                return None
                

            

    model = load_saved_model(dir_pkl, modelname)
    #test = udts_comm()
    #test.init_packet(**kwarg)
    #packet = test.make_packet(verbose=False)
    #client = udts_client(addr=svr_addr,port=svr_port)
    #client.send(label_class)
    while True:
        loc_ipc = os.path.join(dir_code, "formcm204/adq_buffer.pipe")
        adq = ADQ_server(path=loc_ipc)
        adq.run()

        try:
            mfcc = wav_to_mfcc(adq.AI0, len(adq.AI0))
            # for symbol in mfcc:
                # mfcc=np.array(_mfcc).reshape((-1,n_bin*n_frame))

            result = model.predict(mfcc)
            print(result)

            pass
        except Exception as e:
            print(e)
            #socket.close()
            pass


# 
# elif mode=="trans_learning": 
    
#     # model = load_saved_model(dir_pkl, "hybrid_a_0_2_3.h5")
    
#     pass

elif mode=="eval":
    model = load_saved_model(dir_pkl, modelname)
    
    # x, y = build_dataset("x_y")

    loc = "C:/ProgramData/Anaconda3/wav2/dji_inspire2__dji_phantom4/inspire+phantom4.wav"
    data_, sr = librosa.load(
        loc,
        sr=sr,
        mono=False,
        offset=wav_offset,
        res_type='kaiser_best')

    # import soundfile as sf
    data = []

    n_samples = 0

    # try:
    #     data, Fs_ = sf.read(loc)

    # except Exception as e:
    #     print("\t",e,": file, {}".format(loc))
    #     # list_wav.remove(list_wav[j])
    if(np.array(np.shape(data_))[0] == 2):
        data_ = (data_[0,:] + data_[1,:]) / 2
        data_ = data_/np.max(data_) + 1e-16
        pass

    else:
        data_ = data_/np.max(data_) + 1e-16
        pass

    data_ = list(data_)

    if(n_samples < duration*sr*1.2):
        n_samples += len(data_)
        data = data + data_
        data = data[0:int(duration*sr)]
        pass

    else:
        data = data[0:int(duration*sr)]    

    def list_chunk(lst, n):
        return [lst[i:i+n] for i in range(0, len(lst), n)]
    data_chunked = list_chunk(data, sr)
    # data = list(map(float, data))
    data_ = []
    spec = []
    phantom = 0
    import csv
    
    import termios
    import copy
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = copy.deepcopy(old)
    new[3] = new[3] & ~termios.ECHO
    t = '0'
    f = open('Data_logging.csv', 'w', newline = '')
    wr = csv.writer(f)
    while True:
        try:

            termios.tcsetattr(fd, termios.TCSADRAIN, new)
            while True:
                try:

                    if t == '0':
                        print("ambient_noise")
                        wr.writerow([1, 'ambientnoise'])
                    elif t == '1':
                        print("dji_inspire2")
                        wr.writerow([2, 'dji_inspire2'])                        
                    elif t == '2':
                        print("dji_mavic2pro")
                        wr.writerow([3, 'dji_mavic2pro'])                        
                    elif t == '3':
                        print("dji_phantom4")
                        wr.writerow([4, 'dji_phantom4'])
                            
                    elif t == '4':
                        print("Unknown_drone")
                        wr.writerow([5, 'Unknown_drone'])

                    elif t == '5':
                        print("dji_inspire2 dji_mavic2pro")
                        wr.writerow([6, 'dji_inspire2 dji_mavic2pro'])                        
                    elif t == '6':
                        print("dji_inspire2 dji_phantom4")
                        wr.writerow([7, 'dji_inspire2 dji_phantom4'])
                            
                    elif t == '7':
                        print("dji_inspire2 Unknown_drone")
                        wr.writerow([8, 'dji_inspire2 Unknown_drone'])

                    elif t == '8':
                        print("dji_mavic2pro dji_phantom4")
                        wr.writerow([9, 'dji_mavic2pro dji_phantom4'])
                            
                    elif t == '9':
                        print("dji_mavic2pro Unknown_drone")
                        wr.writerow([10, 'dji_mavic2pro Unknown_drone'])
                            
                    elif t == 'a':
                        print("dji_phantom4 Unknown_drone")
                        wr.writerow([11, 'dji_phantom4 Unknown_drone'])

                    pass
                except Exception as e:
                    pass
        except KeyboardInterrupt:
            try:
                t = input('')
            except:
                pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)