from flask_uploads import UploadSet, IMAGES,patch_request_class, AllExcept
from flask import jsonify

upFile = UploadSet('upFile',extensions=AllExcept(()))


def uploadFile(request):
    if request.method == 'POST' and 'videoFile'  in request.files:
      videoName=request.form['fileName']
      request.files['videoFile'].filename=videoName
      filename = upFile.save(request.files['videoFile'])
      return jsonify({"code":200,"message": str(filename)+"上傳成功"})
    elif request.method == 'POST' and 'imageFile'  in request.files:
      imageName=request.form['fileName']
      request.files['imageFile'].filename=imageName
      filename = upFile.save(request.files['imageFile'])
      return jsonify({"code":200,"message": str(filename)+"上傳成功"})
    else:
      return jsonify({"code":404,"message": "NOT FOUND"}),404
