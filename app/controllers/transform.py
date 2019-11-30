from flask import Blueprint, request,jsonify,redirect
import app.modules.transform as transformModule

transformCtrl = Blueprint('transform',__name__)
  

@transformCtrl.route('', methods=['GET'])
def getResult():
    return transformModule.getResult()
  
