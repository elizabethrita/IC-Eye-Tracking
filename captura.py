import cv2
import math
import numpy as np
import os
import pandas as pd
import subprocess
import sys
import time
from gaze_tracking import GazeTracking
import seaborn as sns
sns.set_theme()
import plotly.express as px
import plotly.graph_objects as go
import tkinter as tk
from tkinter import filedialog as fd
from datetime import datetime, timedelta
import json

def default(o):
    if isinstance(o, (timedelta)):
        return o / timedelta(milliseconds=1) # in millis

def get_platform():
    if sys.platform == 'linux':
        try:
            proc_version = open('/proc/version').read()
            if 'Microsoft' in proc_version:
                return 'wsl'
        except:
            pass
    return sys.platform

def open_with_default_app(filename):
    platform = get_platform()
    if platform == 'darwin':
        subprocess.call(('open', filename))
    elif platform in ['win64', 'win32']:
        os.startfile(filename.replace('/', '\\'))
    elif platform == 'wsl':
        subprocess.call('cmd.exe /C start'.split() + [filename])
    else:                                   # linux variants
        subprocess.call(('xdg-open', filename))

def get_video_duration(filename):
    # create video capture object
    data = cv2.VideoCapture(filename)
    # count the number of frames
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(data.get(cv2.CAP_PROP_FPS))
    # calculate duration of the video
    seconds = int(math.ceil(frames / fps))
    return seconds

def capturar():
    print('capturar')
    # create the root window
    root = tk.Tk()
    SCREEN_WIDTH = root.winfo_screenwidth()
    SCREEN_HEIGHT = root.winfo_screenheight()
    root.withdraw()

    filetypes = (
        ('Arquivos de video', '*.avi *.mov *.mp4 *.mpeg *.mpg'),
        ('Todos os arquivos', '*.*')
    )

    #filename = fd.askopenfilename(
    #    title='Abrir um arquivo',
    #    initialdir='/',
    #    filetypes=filetypes)

    print(f'Initializing gaze tracking.')
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    start_delay = 3 # seconds
    print(f'Video will start in {start_delay} seconds.')
    time.sleep(start_delay)
    filename = './assets/video_tcc.mp4'
    open_with_default_app(filename)
    #cv2.setWindowProperty(filename, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    duration = get_video_duration(filename)
    coords = []

    start_time = datetime.now()
    print(f'Main loop started (duration = {duration}).')

    while True:
        
        # We get a new frame from the webcam
        current_time = datetime.now()
        _, frame = webcam.read()
        time_delta = current_time - start_time

        #Quadro de video capturado pelaa webcam redimensionado à resolução da tela
        frame = cv2.resize(frame,(SCREEN_WIDTH, SCREEN_HEIGHT))

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)
        
        annotated_frame = gaze.annotated_frame()
        cv2.imshow('Gaze Tracking', annotated_frame)

        gaze_x_ratio = gaze.horizontal_ratio()
        gaze_y_ratio = gaze.vertical_ratio()
        #print("gaze x", gaze_x_ratio)
        #print("gaze y",gaze_y_ratio)
        if time_delta > timedelta(milliseconds=10):
            
            if gaze_x_ratio != None and gaze_y_ratio != None:

                gaze_x = int((1.0-gaze_x_ratio) * SCREEN_WIDTH)
                gaze_y = int((1.0-gaze_y_ratio) * SCREEN_HEIGHT)

                #print(f'{time_delta}\tx = {gaze_x:.3f} ({gaze_x_ratio:.3f})\ty = {gaze_y:.3f} ({gaze_y_ratio:.3f})')

                coords.append([time_delta, gaze_x, gaze_y])

            if time_delta > timedelta(seconds = duration):
                print(f'Main loop finished after {time_delta}.')
                break

    with open("cache/captura.json", 'w') as f:
        json.dump(coords, f, indent=2, default=default)
    return coords
