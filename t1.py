import cv2
from imwatermark import WatermarkEncoder

bgr = cv2.imread('test.png')
wm = 'test'

encoder = WatermarkEncoder()
WatermarkEncoder.loadModel()
encoder.set_watermark('bytes', wm.encode('utf-8'))
bgr_encoded = encoder.encode(bgr, 'rivaGan')
# bgr_encoded = encoder.encode(bgr, 'dwtDctSvd')
cv2.imwrite('test_wm.png', bgr_encoded)