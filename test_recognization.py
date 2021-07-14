import text_recognize as tr
import glob as gb
import cv2

img_path = gb.glob("./results/\\1_*.jpg")
# print(img_path)
# img_list=[]
for path in img_path:
    image=cv2.imread(path)
    tr.text_detection_spines(image,True)
    print("--------------------")
