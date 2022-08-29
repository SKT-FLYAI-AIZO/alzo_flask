from flask import request,Flask, Response, json, jsonify, make_response
import cv2
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model 
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
from itertools import count
import gc
from datetime import date, datetime



app = Flask(__name__)

model_path = "./weights/weights.h5"
model = load_model(model_path)
    
fps = 30
sec = 4

def save_shorts(video_path, shorts_unique, start_time,upload_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c = 3    

    segRange = [(shorts_idx[0], shorts_idx[0] + fps*sec*2) for shorts_idx in shorts_unique]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    mk_file = []
    for idx,(begFidx,endFidx) in enumerate(segRange):
        writer = cv2.VideoWriter(f'{upload_path[:-4]}_{idx}.mp4',fourcc,fps,(w, h))
        mk_file.append(f'{upload_path[:-4]}_{idx}.mp4')
        cap.set(cv2.CAP_PROP_POS_FRAMES,begFidx)
        ret = True # has frame returned
        time_ = 0
        time_list = []
        while(cap.isOpened() and ret and writer.isOpened()):
            ret, frame = cap.read()
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            time_ = np.datetime64(start_time) + np.timedelta64(int(1000 * frame_number / fps), 'ms')
            cv2.putText(frame, str(time_)[:-3], org=(10, 80), fontFace=1, fontScale=5, color=(0, 0, 0), thickness=4)

            if frame_number < endFidx:
                writer.write(frame)
            else:
                break
        time_list.append(time_)
        writer.release()
    return mk_file,time_list

def shorts_intersect(shorts_idx):
    shorts_unique = [shorts_idx[0]]
    for i in range(1, len(shorts_idx)):
        intersect = np.intersect1d(shorts_unique[-1], shorts_idx[i])
        if len(intersect) >= 1:
            continue
        else:
            shorts_unique.append(shorts_idx[i])

    return shorts_unique

def get_shorts(pred_lst):

    shorts_len = fps * sec
    shorts_idx = []
    for idx in range(len(pred_lst) - shorts_len):
        shorts = pred_lst[idx:idx+shorts_len].tolist()
        
        if shorts.count(1) >= 30:
            shorts_idx.append(range(idx, idx+shorts_len*2))
    
    if len(shorts_idx):
        shorts_unique = shorts_intersect(shorts_idx)
        print(shorts_unique)
        return shorts_unique
    else:
        return None

def video_show(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c = 3
    
    threshold=0.95
    pred_lst = []
    tmp = 0
    batch = np.empty((1, 512, 512, 3))
    batch_size = 4

    while True:
        ret, frame = cap.read()
        print("Predicting")
        if ret:
            img = frame.copy()
            img = cv2.resize(img, (512, 512))
            img = img.astype(np.float32)
            img /= 255.
            img = np.expand_dims(img, axis=0)
            if tmp:
                if (tmp % batch_size) != 0:
                    batch = np.concatenate((batch, img), axis=0)
                else:
                    batch = np.concatenate((batch, img), axis=0)
                    y_pred = model.predict(batch[1:], verbose=0,batch_size=batch_size)
                    batch = np.empty((1, 512, 512, 3))
                    y_pred = (np.array(y_pred)[:, 1] > threshold)            
                    pred_lst.append(y_pred)
                tmp += 1

            else:
                tmp += 1
                continue
            
        else:
            break

        #cv2.putText(frame, violation[y_pred], org=(10, 80), fontFace=1, fontScale=5*(y_pred+1), color=color_list[y_pred], thickness=4)
        #cv2.imshow("frame", frame)
        #cv2.waitKey(1)
    pred_lst = np.reshape(pred_lst, (-1, ))

    return pred_lst

def predict_violation(frame, model):
    img = frame.copy()
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32)
    img /= 255.
    img = np.expand_dims(img, axis=0)

    y_pred = model.predict(img, verbose=0)
    
    return np.argmax(y_pred)      # 0: normal, 1: violation



@app.route('/been',methods=['POST'])
def predict_play():
    print("start")
    data = {'path':'/test/path.txt'
            ,'lat':'80.12'
            ,'lon':'190.1'
            ,'time':'2022-02-12 20:20:20'}
    response = app.response_class(
        response=json.dumps(data),
        mimetype='application/json'
    )
    return response

@app.route('/play',methods=['POST'])
def temp():
    params = request.get_json()
    # 본 서버용
    download_file_path = './temp/'+params['path']
    connect_str = os.getenv("STORAGE_CONNECTION_STRING")    
    print(connect_str)
    # 내부 테스트용
    #connect_str = 'DefaultEndpointsProtocol=https;AccountName=blobtestyummy;AccountKey=RKYhaSjt1AFoiVOFU6p/63bnDCIc95yD9+w46YSA1pCC/rTUbBf+pCHNlD6eBewhKbEmlFv5mfeV+AStBlsy3Q==;EndpointSuffix=core.windows.net'
    
    # download_file_path = './media/'+params['path']
    # date = params['date']


    download_container = os.getenv('STORAGE_AZURE_CONTAINER')
    upload_container = os.getenv('STORAGE_CROPPED_CONTAINER')
    print(upload_container)
    print(download_container)

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(download_container)
    #container_client = blob_service_client.get_container_client(download_container)
    data = {'upload': 'fail'}
    try:
        with open(download_file_path, 'wb') as download_file:
            download_file.write(container_client.download_blob(params['path']).readall())
            download_file.close()
            data['upload']='succeced'
    except:
        #print("여기 도착")
        return make_response(jsonify(data), 503)
    
    # start = time.time()
    upload_file_path = './media/'+params['path']
    video_path = download_file_path
    
    # # print(video_path)
    pred_lst = video_show(video_path, model)
    # # print("t"+str(pred_lst))
    shorts_unique = get_shorts(pred_lst)
    mk_file_list = []
    gps_time_list =[]
    if shorts_unique is not None:
        mk_file_list, gps_time_list = save_shorts(video_path, shorts_unique,params['time'],upload_file_path)
    
    for file_path in mk_file_list:
        file_name = file_path.split('/')
        print(file_name[-1])
        temp = file_name[-1]
        blob_client = blob_service_client.get_blob_client(container=upload_container,blob=temp)
        with open(file_path, 'rb') as data:
            blob_client.upload_blob(data)
            data.close()
    
    for file_path in mk_file_list:
        os.remove(file_path)
    for i in len(mk_file_list):
        mk_file_list[i] = "https://aizostorage.blob.core.windows.net/aizo-cropped/"+ mk_file_list[i]
    
   ## GPS, time 시간 처리 미구현
#    다운로드 링크 받아서 처리해줘야함
#    with open()
    os.remove(download_file_path)
    data = {
        "path" : mk_file_list,
        #"gps" : [lat,lon]
    }
    return make_response(jsonify(data), 201)

if __name__== '__main__':
    app.run(debug = True, port=8080)
