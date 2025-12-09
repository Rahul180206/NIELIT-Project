from earphone_workflow import EarphoneWorkflow
import cv2

ew = EarphoneWorkflow()  # reads .env
img = cv2.imread("sample.jpg")  # use sample image in project root
dets = ew.detect(img)
print("Earphone detections:", dets)
