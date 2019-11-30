from flask import Blueprint, request,jsonify,redirect
import app.modules.transform as transformModule

transformCtrl = Blueprint('transform',__name__)
  

@transformCtrl.route('', methods=['POST'])
def getResult():
    insertValues = request.get_json()
    print(insertValues) #request.get_json()['videoName']
    videoName=request.get_json()['videoName']
    return transformModule.getTransform(videoName)
  
