from __future__ import absolute_import, division, print_function, unicode_literals


from udts_sock import *
import argparse
import os
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa
from tqdm import trange
from ast import literal_eval
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, utils
import mlpy
from sklearn.model_selection import train_test_split
import glob
import cv2
import soundfile as sf
import PIL.Image as Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_score
import re
from scipy.io import wavfile

print(('TensorFlow version: {0}').format(tf.__version__))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
      
    print(e)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
      
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
      
    print(e)





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
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')

parser.add_argument(
    '-i', '--interval', type=float, default=1,
    help='minimum time between plot updates (default: %(default)ms)')

parser.add_argument(
    '-sr', '--samplerate', type=float, default=48000,
    help='sampling rate of the input')

parser.add_argument(
    '-n', '--downsample', type=int, default=1, metavar='N',
    help='display every Nth sample (default: %(default))')

parser.add_argument(
    "-c", "--clear_dataset", type=str, default='False', metavar="ex) True",
    help = "reset the dataset pickle")

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

parser.add_argument(
    "-td", "--training_dir", type = str, default = "training_wav_color_2",
    help = "input training (default: %(default))")

parser.add_argument(
    "-ts", "--time_samples", type = int, default = 16000,
    help = "time_samples is , default:%(default)")

args = parser.parse_args(remaining)

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
channels = [c for c in range(args.channels+1)[1:]]
mapping = [c - 1 for c in channels]
mode = args.mode
lr = args.learning_rate
epochs = args.n_epoch
batch_size = args.n_batch
len_seq = args.len_seq
source_dir2 = args.training_dir
time_samples = args.time_samples


dir_code = os.getcwd()
dir_wav = os.path.join(dir_code, source_dir)

os.chdir(dir_wav)

label_class = os.listdir()

def isWaveFile(name):
    return True if(name.find(".wav") > -1 or name.find(".WAV") > -1) else False

def isValidDir(loc):
    return True if(os.path.isdir(loc)) else False

label_class = [s for s in label_class if(isValidDir(s) and not isWaveFile(s))]

if(isWaveFile("wav2")):
    label_class = ["single_file_test",]
    n_class = 1

else:
    os.chdir(dir_wav)
    label_class = os.listdir()
    label_class = [s for s in label_class if(isValidDir(s) and not isWaveFile(s))]

if(label_class != []):
    n_class = len(label_class)

else:
    print("check your data file structure")
    exit(1)

def load_wav_dir(dir_wav, n_class):
    global sr
    data = []


    if(isValidDir(dir_wav)):
        list_wav = os.listdir(dir_wav)
        list_wav = [s for s in list_wav if (".wav" in s) or (".WAV" in s)]        
        n_files = len(list_wav)
        list_wav = [os.path.join(dir_wav, s) for s in list_wav]
        sr_list = [librosa.get_samplerate(s) for s in list_wav]

    elif(os.path.isfile(dir_wav)):
        file_wav = [dir_wav]

    n_samples = 0


    for j in trange(n_files, desc="loading class %s..." % n_class):
        
        loc = os.path.join(dir_wav, list_wav[j])



        try:
            data, Fs_ = sf.read(loc)

        except Exception as e:
            print("\t",e,": file, {}".format(loc))
            pass

    return Fs_, data

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def cwt(data, folder):

    data = np.array(data).flatten()
    data_chunked = list_chunk(data, time_samples)
    
    try:
        for i in range(0, len(data_chunked)):
            cwt_data = np.array(data_chunked[i], dtype = float)
        
            omega0 = 6
            scales = mlpy.wavelet.autoscales(N = len(cwt_data), dt = 1, dj = 0.01, wf = 'morlet', p = omega0)
            scales = scales[200:1400]
            spec = mlpy.wavelet.cwt(cwt_data, dt = 1, scales = scales, wf = 'morlet', p = omega0)
            
            name = "_data.jpg"
            
            figure = plt.figure()

            plt.imshow(abs(spec), interpolation = 'bilinear', cmap = 'turbo', aspect = 'auto')
            plt.savefig(folder + str(i) + name)

            plt.close(figure)



    except Exception as e:
        print(e)

def build_dataset(mode="slices", train=.7, test=.3):
    label_file = []
    n_files = []
    Fs = []
    w_data = []
    for i in range(n_class):
        loc = os.path.join(dir_wav, label_class[i])
        _files = os.listdir(loc)
        _files = [s for s in _files if not ".ini" in s]
        label_file.append(_files)
        n_files.append([label_class[i], len(_files)])

    for i, _class in zip(range(n_class), label_class):
        loc = os.path.join(dir_wav, _class)
        if (len(os.listdir(loc))>0):
            Fs_, data_ap = load_wav_dir(loc, _class)
            Fs.append(Fs_)
            w_data.append(data_ap)
            n_samples.append(len(w_data))
        dist = list(np.asarray(n_samples)/sum(n_samples))
    return Fs, w_data


n_samples = []
label_file = []
n_files = []


spec = []
wav_loc = os.path.join(dir_code, source_dir2)

X = []
Y = []

image_w = 64
image_h = 96



def load_saved_model(loc=dir_code, name="model_10.h5"):
    file = os.path.join(loc, name)
    try:
        model = models.load_model(file)
        print("\tModel %s has been loaded."%loc)
        pass
    except Exception as e:
        print("\t Loading model %s canceled. Check the model file first.", loc)
        pass
    
    return model

def fig2data(fig):

    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    img = Image.frombytes("RGB", (w, h), buf.tostring())
    
    img = np.asarray(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
    
def cwt_stream(ADQ_data):
    start = time.time()
    ADQ_data = np.array(ADQ_data)
    data = np.array(ADQ_data).flatten()
    data_chunked = list_chunk(data, time_samples)
    cwt_data = np.array(data_chunked[0], dtype = float)
    omega0 = 6
    scales = mlpy.wavelet.autoscales(N = len(cwt_data), dt = 1, dj = 0.01, wf = 'morlet', p = omega0)
    scales = scales[200:1400]
    spec = mlpy.wavelet.cwt(cwt_data, dt = 1, scales = scales, wf = 'morlet', p = omega0)

    figure = plt.figure()
    
    plt.imshow(abs(spec), interpolation = 'bilinear', cmap = 'turbo', aspect = 'auto')
    img = fig2data(figure)

    plt.close(figure)

    img = cv2.resize(img, None, fx = image_h/img.shape[1], fy = image_w/img.shape[0])
    img = np.array(img)


    X = []
    X.append(img/255)

    X = np.array(X)
    print("time : ", time.time() - start)

    return X

modelname = "/model_9.h5"

if mode=="training" :


    n_samples = []
    label_file = []
    n_files = []

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

    Fs, data = build_dataset("slices")
    spec = []
    wav_loc = os.path.join(dir_code, source_dir2)
    for i in range(0, len(data)):
        try:
            os.makedirs(os.path.join(wav_loc, label_class[i]))
        except:
            print("The folder already exists. ")
        folder = wav_loc + "/" + label_class[i] + "/"
        cwt(data[i], folder)

    X = []
    Y = []
    
    for subset in ('train', 'test'):
        path_to_subset = f'C:/ProgramData/Anaconda3/training_wav_color_2'
        
        for folder in os.listdir(path_to_subset):
            for image in os.listdir(os.path.join(path_to_subset, folder)):
                path_to_image = os.path.join(path_to_subset, folder, image)
                image = cv2.imread(path_to_image)
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = cv2.resize(image, None, fx = image_h/image.shape[1], fy = image_w/image.shape[0])
                # image = np.expand_dims(image, axis=2)
                label = re.findall(r'\w+\w+', folder)[0].split('__')
                image = image/255
                X.append(image)
                Y.append(label)




    X = np.array(X)
    Y = np.array(Y)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    

    model = tf.keras.Sequential()
    input_shape = (image_w, image_h, 3)

    model.add(layers.Conv2D(32, kernel_size = (3, 3), padding = 'same', input_shape = input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size = (2, 2), padding = 'same'))

    model.add(layers.Conv2D(32, kernel_size = (3, 3), padding = 'same'))
    model.add(layers.MaxPooling2D(pool_size = (4, 2), padding = 'same'))

    model.add(layers.Conv2D(64, kernel_size = (3, 3), padding = 'same'))
    model.add(layers.MaxPooling2D(pool_size = (4, 1), padding = 'same'))

    model.add(layers.Conv2D(64, kernel_size = (3, 3), padding = 'same'))
    model.add(layers.MaxPooling2D(pool_size = (2, 1), padding = 'same'))

    model.add(layers.Reshape(target_shape=(-1,64)))

    model.add(layers.Bidirectional(layers.GRU(24)))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(32, activation='relu'))
    
    model.add(layers.Dense(len(mlb.classes_), activation = 'sigmoid'))

    model.summary()

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    history = model.fit(X_train, Y_train, batch_size = 128, validation_data = (X_test, Y_test), epochs = 50, verbose = 1)

    model.save('model_10.h5')

    print(mlb.classes_)


    pass

elif mode=="stream":

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


    model = tf.keras.models.load_model(dir_code+modelname)
    test = udts_comm()
    test.init_packet(**kwarg)
    packet = test.make_packet(verbose=False)
    client = udts_client(addr=svr_addr,port=svr_port)
    client.send(label_class)
    drone_dect = 0

    while True:
        loc_ipc = os.path.join(dir_code, "formcm204/adq_buffer.pipe")
        adq = ADQ_server(path=loc_ipc)
        adq.run()

        try:
            cwt_data = cwt_stream(adq.AI0)
            result = model.predict(cwt_data)
            print(result)
            background_threshold = 1e-1
            np_result = np.array(result[0])
            drone_background = np_result.mean(axis = 0)

            if(max(drone_background) == drone_background[1]):
                second_softmax = set(drone_background)
                second_softmax.remove(max(drone_background))

                if(max(second_softmax) < background_threshold):
                    print("There is no drone!")
                    drone_dect = 0

                else:
                    drone_dect = 1

            else:
                drone_dect = 1
                pass
            pass

        except Exception as e:
            print(e)
            pass


elif mode == "eval":

    loc = "C:/ProgramData/Anaconda3/wav2_many_label/3dr_solo__intel_rtf/3dr_intel.wav.wav"
    data = []
    n_samples = 0

    try:
        data, Fs_ = sf.read(loc)

    except Exception as e:
        print("\t",e,": file, {}".format(loc))
        pass


    data = np.array(data).flatten()
    data_chunked = list_chunk(data, time_samples)
 
    spec = []
    dir_code = os.path.join(dir_code, source_dir)

    model = tf.keras.models.load_model(dir_code + modelname)

    for i in range(0, len(data_chunked)):
        cwt_data = np.array(data_chunked[i], dtype = float)

        cwt_data = cwt_stream(cwt_data)
        result = model.predict(cwt_data)
        print(result[0])