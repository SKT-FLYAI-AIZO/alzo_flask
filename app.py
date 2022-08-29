from flask import request,Flask, Response, json, jsonify, make_response
import cv2
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model 
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__



app = Flask(__name__)

azure_info = {'account_name' : 'aizost',
               'account_key' : 'CKwlWVDRI8l0ih7HQZetCLaSrX4PaHUeo0lmzOFYhmkLadmy9/sHoueOoLXwbDvNh7jjC46OHudf+AStNsLWeg==',
               'azure_container' : {'original ' : 'aizo-container','edit' :  'aizo-cropped'} ,
               'blob_storage_connect_string':'DefaultEndpointsProtocol=https;AccountName=aizost;AccountKey=CKwlWVDRI8l0ih7HQZetCLaSrX4PaHUeo0lmzOFYhmkLadmy9/sHoueOoLXwbDvNh7jjC46OHudf+AStNsLWeg==;EndpointSuffix=core.windows.net'
             }

wieght_path = "./weights/weights.h5"

violation = {
    0: 'NORMAL',
    1: 'VIOLATION'
}

fps = 30
sec = 4

color_list = [(0, 255, 0), (0, 0, 255)]    # 정상: 초록, 위반: 빨강

def save_shorts(video_path, shorts_unique, start_time):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c = 3    

    segRange = [(shorts_idx[0], shorts_idx[0] + fps*sec*2) for shorts_idx in shorts_unique]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    for idx,(begFidx,endFidx) in enumerate(segRange):
        writer = cv2.VideoWriter(f'./temp/{bolb_name}{idx}.mp4',fourcc,fps,(w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES,begFidx)
        ret = True # has frame returned
        while(cap.isOpened() and ret and writer.isOpened()):
            ret, frame = cap.read()
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            time_ = start_time + np.timedelta64(int(1000 * frame_number / fps), 'ms')
            cv2.putText(frame, str(time_)[:-3], org=(10, 80), fontFace=1, fontScale=5, color=(0, 0, 0), thickness=4)

            if frame_number < endFidx:
                writer.write(frame)
            else:
                break
        writer.release()


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
    with open('./incheon/incheon.txt', 'w') as f:
        for idx in range(len(pred_lst) - shorts_len):
            shorts = pred_lst[idx:idx+shorts_len].tolist()
            
            f.write(f'{shorts.count(1)}\n')
            if shorts.count(1) >= 45:
                shorts_idx.append(range(idx, idx+shorts_len*2))
        f.close()
        
        if len(shorts_idx):
            shorts_unique = shorts_intersect(shorts_idx)
            print(shorts_unique)
            return shorts_unique
        else:
            return None

"""def video_show_test(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c = 3
    
    pred_lst = []
    batch = np.empty((1, 512, 512, 3))
    batch_size = 8
    tmp = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        
        if ret:
            if tmp % 3 == 0:
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
                        y_pred = model.predict(batch[1:], verbose=0)
                        batch = np.empty((1, 512, 512, 3))
                        gc.collect()
                    #y_pred = predict_violation(frame, model)
                        pred_lst.append(np.argmax(y_pred, axis=1))
                tmp += 1
            else:
                tmp += 1
                continue
            
        else:
            break
    print(f'time: {time.time() - start}')
        #cv2.putText(frame, violation[y_pred], org=(10, 80), fontFace=1, fontScale=5*(y_pred+1), color=color_list[y_pred], thickness=4)
        #cv2.imshow("frame", frame)
        #cv2.waitKey(1)
    pred_lst = np.reshape(pred_lst, (-1, ))
    return pred_lst"""

def video_show(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c = 3
    
    pred_lst = []
    tmp = 0
    batch = np.empty((1, 512, 512, 3))
    batch_size = 8

    while True:
        ret, frame = cap.read()
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
                    y_pred = model.predict(batch[1:], verbose=0)
                    batch = np.empty((1, 512, 512, 3))
                    gc.collect()
                #y_pred = predict_violation(frame, model)
                    pred_lst.append(np.argmax(y_pred, axis=1))
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



@app.route('/play/yummy',methods=['POST'])
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

@app.route('/yummy',methods=['POST'])
def temp():
    params = request.get_json()
    # 본 서버용
    # data = params['date']
    # download_file_path = './'+params['path']
    # connect_str = os.getenv("STORAGE_CONNECTION_STRING")    
    # 내부 테스트용
    connect_str = 'DefaultEndpointsProtocol=https;AccountName=blobtestyummy;AccountKey=RKYhaSjt1AFoiVOFU6p/63bnDCIc95yD9+w46YSA1pCC/rTUbBf+pCHNlD6eBewhKbEmlFv5mfeV+AStBlsy3Q==;EndpointSuffix=core.windows.net'
    
    download_file_path = './media/'+params['path']
    # date = params['date']

    # print(2)
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # container_client = blob_service_client.get_container_client(os.getenv('STORAGE_ACCOUNT_NAME'))
    container_client = blob_service_client.get_container_client('blob')
    data = {'upload': 'fail'}
    try:
        with open(download_file_path, 'wb') as download_file:
            download_file.write(container_client.download_blob(params['path']).readall())
            download_file.close()
            data['upload']='succeced'
    except:
        return make_response(jsonify(data), 503)
    
    # model = load_model(model_path)
    # start = time.time()
    # #BLOBNAME = '1_Busan_2022-08-23.mp4'
    # #download_file_path = blob_read(BLOBNAME)
    # video_path = download_file_path
    # # print(video_path)
    # pred_lst = video_show(video_path, model)
    # # print("t"+str(pred_lst))
    # shorts_unique = get_shorts(pred_lst)
    # if shorts_unique is not None:
    #     save_shorts(video_path, shorts_unique)
    # # print(f'time: {time.time() - start}s')   

    # for file_path in mk_file_list:
    #     blob_client = blob_service_client.get_blob_client(container='blob', blob=file_path)
    #     with open(os.path.join('./media',file_path), 'rb') as data:
    #         blob_client.upload_blob(data)
   
#    다운로드 링크 받아서 처리해줘야함
#    with open()
    os.remove(download_file_path)
    return make_response(jsonify(data), 201)
    

    

# @app.route('/action',methods = ['POST'])
# def test_method():
#     if not request.is_json :
#         return Response({'state': 'Not Json'},status=404,mimetype='application/json')
#     params = request.get_json()
#     if params['test'] == 0:
#         return Response({'state': 'Not video'},status=404,mimetype='application/json')
#     else:    
#         blob_service_client = BlobServiceClient.from_connection_string(azure_info['blob_storage_connect_string'])
#         container_client = blob_service_client.get_container_client('aizo-cropped')
#         file_path = 'media\flask_blobupload_test.mp4'

#         blob_client = blob_service_client.get_blob_client(container='aizo-cropped',blob=file_path)

#         with open(file_path, "rb") as data:
#             blob_client.upload_blob(data)

#         return Response({'state':'Upload_finish'},status=201,mimetype='application/json')
if __name__== '__main__':
    app.run(debug = True, port=8080)
