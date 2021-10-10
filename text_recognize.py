  
from imutils.object_detection import non_max_suppression
import imutils
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from crnn import demo
from spellchecker import SpellChecker
spell = SpellChecker()

# f = open("recognized.txt", "a+")

def listToString(s): 
    
    # initialize an empty string
    str1 = " " 
    
    # return string  
    return (str1.join(s))

def get_bounding_boxes(orig_image, boxes, rW, rH, padding, origW, origH,debug=False):
    # results = []
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    texts=[]
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the image
        #cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))

        # extract the actual padded ROI
        roi = orig_image[startY:endY, startX:endX]

        if(endX - startX < endY - startY):
            roi = imutils.rotate_bound(roi, 90)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        text = demo.text_recognize(roi)
        corrected = spell.unknown([text])
        for word in corrected:
            text=spell.correction(word)
        if(len(text)>0):
            texts.append(text)
        # results.append(((startX, startY, endX, endY),text))
    if(debug):
        print(*texts)
    tex=listToString(texts)
    tex=tex.strip()
    # if(len(tex)>0):
    #     f.write(tex+"\n")
    # results = sorted(results, key=lambda r:r[0][1])
    # print(results)
    return tex


def decode_predictions(scores, geometry, min_confidence):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < min_confidence:
                continue
                
            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            
            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            
            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
            
    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


def forward_pass(image, net, W, H, debug = False):
    
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
     
    # show timing information on text prediction
    if(debug):
        print("[INFO] text detection took {:.6f} seconds".format(end - start))

    return (scores, geometry)

def draw_bounding_boxes(image, results):
	output = image.copy()

	# loop over the results
	for ((startX, startY, endX, endY),text) in results:
		cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)

		
	return output

def text_detection(image,net, debug=False):
    min_confidence = 0.5
    width = 768
    height = 128
    padding = 0.02

    orig = image.copy()
    (origH, origW) = image.shape[:2]

    (newW, newH) = (width, height)
    rW = origW / float(newW)
    rH = origH / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    scores, geometry = forward_pass(image, net, W, H)

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry, min_confidence)

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    results = get_bounding_boxes(orig, boxes, rW, rH, padding, origW, origH, debug)

    

    if(debug==True):
        proc_img = draw_bounding_boxes(orig, results)
        cv2.imshow("boxed",proc_img)
        # cv2.imwrite("boxed.jpg",proc_img)
        cv2.waitKey()
    return results

def text_detection_spines(spine_image, debug=False):
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    # for spine_image in spine_images:
    image= imutils.rotate_bound(spine_image,-90)
    tex= text_detection(image,net,debug)
    return tex

# image=cv2.imread("./results/1_2.jpg")
# img_list=[]
# img_list.append(image)
# text_detection_spines(img_list)