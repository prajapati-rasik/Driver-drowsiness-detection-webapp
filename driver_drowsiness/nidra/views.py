from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import cv2
import base64
import numpy as np
from nidra.predictor import Predictor

# Create your views here.
def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def developer(request):
    return render(request, 'developers.html')

def detection_api(request):
    return render(request, 'camera.html')

@csrf_exempt
def video_feed_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            image_data = data.get("image")

            # Convert base64 to image
            format, imgstr = image_data.split(';base64,')
            img_bytes = base64.b64decode(imgstr)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Create prediction object
            prediction = Predictor.predict(img)

            return JsonResponse({"prediction": prediction})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request"}, status=400)
