import sys
import pandas as pd
import os

DATA_PATH = './data/train'

def audio_link(id):
    return 'https://www.youtube.com/watch?v=' + id

def download(link, name, loc=None, sr=16000):
    command = 'cd %s;' % loc
    command += 'youtube-dl -x --audio-format wav -o o' + name + '.wav ' + link + ';'
    command += 'ffmpeg -i o%s.wav -ar %d -ac 1 %s.wav;' % (name,sr,name)
    command += 'rm o%s.wav' % name
    os.system(command)

def cut(name, start, end):
    length = end - start
    command = 'cd %s;' % DATA_PATH + 'train/'
    command += 'sox %s.wav trim_%s.wav trim %s %s;' % (name ,name, start, length)
    command += 'rm %s.wav' % name
    os.system(command)    


# def audio(data, loc):

#     for i in range(9999, 20000):
#         name = 'trim_train' + str(i)
#         link = link(data.loc[i, 'link'])
#         start_time = data.loc[i, 'start_time']
#         end_time = data.loc + 3.0 # only 3 seconds
#         download(loc, name, link)
#         cut(loc, name, start_time, end_time) 

#         download(name, link)

# data = pd.read_csv('./avspeech_train.csv')
# audio(data, '_train')

download('https://www.youtube.com/watch?v=6ml9RQaqbF0&ab_channel=stomachandintestinessound/',
         name='bowel_sound2',
         loc='/Users/sehwan/Desktop/datasets/bowel_sound/',
         )