import os
import cv2
from time import time
from decord import VideoReader
from decord import cpu, gpu
import numpy as np



video_filename='/srv/essa-lab/flash3/vcartillier3/egoexo-view-synthesis/data/egoexo4d/takes/georgiatech_covid_18_2/frame_aligned_videos/downscaled/448/cam01.mp4'
frame_idx = 1000


def opencv_reader(filename, frame_idx):
    t0 = time()
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    t1 = time()
    return frame, t1-t0

def decord_reader(filename, frame_idx):
    t0 = time()
    vr = VideoReader(filename, ctx=cpu(0))
    frame = vr[frame_idx]
    frame = frame.asnumpy()
    t1 = time()
    return frame, t1-t0


frame_cv, time_cv = opencv_reader(video_filename,frame_idx)
frame_dc, time_dc = decord_reader(video_filename,frame_idx)


diff = np.mean(np.abs(frame_cv - frame_dc))


print(" ## Opencv reading time: ", time_cv)
print(" ## Decord reading time: ", time_dc)
print(" ## diff reading: ", diff)



