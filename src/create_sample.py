import cv2
import numpy as np
import urllib.request
import os

os.makedirs('data', exist_ok=True)
url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
print(f"Downloading {url}...")
req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, -1)
h, w, _ = img.shape
out = cv2.VideoWriter('data/sample.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

for i in range(30):
    M = np.float32([[1, 0, np.sin(i * 0.5) * 5], [0, 1, np.cos(i * 0.5) * 5]])
    shifted = cv2.warpAffine(img, M, (w, h))
    out.write(shifted)
out.release()
print("Created data/sample.mp4")
