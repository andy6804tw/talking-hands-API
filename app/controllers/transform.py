from flask import Blueprint, request,jsonify,redirect
import app.modules.transform as transformModule

transformCtrl = Blueprint('transform',__name__)
  

@transformCtrl.route('/video', methods=['POST'])
def getResult():
    insertValues = request.get_json()
    print(insertValues) #request.get_json()['videoName']
    videoName=insertValues['videoName']
    modelIdx=int(insertValues['modelIdx'])
    return transformModule.getTransform(videoName, modelIdx)
  

@transformCtrl.route('/image', methods=['POST'])
def getImgTransform():
    insertValues = request.get_json()
    srcImage=insertValues['srcImage']
    modelIdx=int(insertValues['modelIdx'])
    return transformModule.imgTransform(srcImage, modelIdx)