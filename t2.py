import cv2
from imwatermark import WatermarkDecoder

bgr = cv2.imread('test_wm.png')

decoder = WatermarkDecoder('bytes', 32)
WatermarkDecoder.loadModel()
# watermark = decoder.decode(bgr, 'dwtDctSvd')
watermark = decoder.decode(bgr, 'rivaGan')
print(watermark.decode('utf-8'))