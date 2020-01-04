import cv2
import io
import socket
import struct
import time
import pickle
import zlib


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect(('0.tcp.ngrok.io', 12698))
client_socket.connect(('127.0.0.1', 8485))
connection = client_socket.makefile('wb')

cam = cv2.VideoCapture(0)

# cam.set(3, 128)
# cam.set(4, 128)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    ret, frame = cam.read()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
#    data = zlib.compress(pickle.dumps(frame, 0))
    print(frame.shape)
    # frame=frame[:,::-1,:]
    data = pickle.dumps(frame, 0)
    size = len(data)


    # print("{}: {}".format(img_counter, size))
    if img_counter%8==0:
        client_socket.sendall(struct.pack(">L", size) + data)
    img_counter += 1
    # print(img_counter)
    response = client_socket.recv(1024).decode()
    print(response)

cam.release()