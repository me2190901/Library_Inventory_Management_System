import cv2
import os
import sys
import math
import imutils
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate



    

def canny_edge(img, low_th, high_th):
    edged = cv2.Canny(img, low_th, high_th)

    return edged


def gaussian_blur(img, filter_size, sigma):
    proc_img = cv2.GaussianBlur(img, (filter_size, filter_size), sigma)

    return proc_img


def global_thresholding(img, th_value, max_value):
    thresh_img = cv2.threshold(img, th_value, max_value, cv2.THRESH_BINARY)

    return thresh_img



def connected_components(img):
    '''
    Finds all connected components in a binary image and assigns all connections
    within a component to a unique value for that component.
    Returns the processed image, and the values of the unique components.

    '''
    levels, proc_img = cv2.connectedComponents(img, connectivity = 8)

    return proc_img, levels


def remove_short_clusters(img, levels, th = 200):
    hist = []

    for i in range(levels):
        hist.append(0)

    for i in range(len(img)):
        for j in range(len(img[i])):
            hist[img[i][j]] += 1

    #th = 200
    max_freq = []

    #new = np.array(img)
    new_img = np.zeros(img.shape)
    np.copyto(new_img, img)

    for i in range(1, levels):
        if(hist[i] > th):
            max_freq.append(i)
            #new_img[img == i] = 255

    for l in max_freq:
        new_img[img == l] = 255

    for i in range(len(new_img)):
        for j in range(len(new_img[i])):
            if(new_img[i][j] != 255):
                new_img[i][j] = 0
    return new_img


def clip_line(x1, y1, x2, y2, r, c):
    y1 = -y1
    y2 = -y2
    
    if(x2-x1 == 0):
        return (x1, 0, x2, r-1, 90)
    
    m = (y2 - y1) / (x2 - x1)
    theta = math.degrees(math.atan(m))
    
    x1_new = x1 - (y1 / m)
    x2_new = x1 + (-r - y1) / m
    
    return int(x1_new), 0, int(x2_new), r-1, theta


def apply_hough_transform(img, image, min_votes, debug = False):
    lines = cv2.HoughLines(img.astype('uint8'), 1, np.pi/180, min_votes)
    r, c = img.shape

    output = image.copy()
    
    all_theta = []
    actual_theta = []
    points = []
    temp_points = []

    for values in lines:
        rho, theta = values[0]
        all_theta.append(math.degrees(theta))
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        x3, y3, x4, y4, t = clip_line(x1, y1, x2, y2, r, c)
        actual_theta.append(t)

        #if(t > 65 or t < -65):
        temp_points.append([(x3, y3), (x4, y4), t])

    theta_values = {}
    for theta in actual_theta:
        theta = np.abs(int(theta))
        if(theta in theta_values.keys()):
            theta_values[theta] += 1
        else:
            theta_values[theta] = 1


    dominant_dir = max(theta_values, key=theta_values.get)

    for p in temp_points:
        (x3, y3), (x4, y4), t = p

        if(np.abs(int(t)) < dominant_dir + 10 and np.abs(int(t)) > dominant_dir - 10):
            points.append([(x3, y3), (x4, y4), t])
            cv2.line(output, (x3,y3), (x4,y4), (0,255,255), 2)

    return points


def merge_lines2(points,image):
    points.sort(key = lambda point: point[0][0])
    points = points
    image_height=points[0][1][1]
    new_points=[]
    prev=points[0][0][0]
    new_points.append([points[0][0][0],image_height])
    near=1
    start=points[0][0][0]
    for i in range(1,len(points)):
        if(points[i][0][0]-start>30):
            new_points.append([points[i][0][0],image_height])
            start=points[i][0][0]
            near=1
        else:
            new_points[-1][0]=int((start+points[i][0][0])/2)
            
    output=image.copy()
    for p in new_points:
        cv2.line(output, (int(p[0]), 0), (int(p[0]), p[1]), (0, 255, 255), 2)
    # cv2.imshow("merge2",output)
    return new_points, output



img_width = 600
img_height = 450


# --------------Hough Transfrom--------
def pre_process(img):
    img_resized = cv2.resize(img, (img_width, img_height))
    img_blur = cv2.GaussianBlur(img_resized, (3, 3), 0)
    return img_resized, img_blur


def draw(img, lines):
    new_img = img.copy()
    for rho, theta in lines[:]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(new_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return new_img


def draw_vertical(img, lines):
    new_img = img.copy()
    
    for rho, theta in lines[:]:
        x1 = 100
        x2 = 0
        y1 = 0
        y2 = img_width
        print(x1,x2,y1,y2)
        cv2.line(new_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return new_img


def line_reduce(lines):
    i = 0
    j = 0
    lines_final = []
    while i < len(lines) - 1:
        if j >= len(lines) - 1:
            break
        j = i + 1
        lines_final.append(lines[i])
        while j < len(lines) - 1:
            if lines[j][0] - lines[i][0] > 10:
                i = j
                break
            else:
                j = j + 1
    return lines_final


def line_sifting(lines_list):
    lines = []
    for rho, theta in lines_list[:]:
        if (theta < (np.pi / 6.0)) or (theta > (11 * np.pi / 6.0)) or ((theta > (5 *np.pi / 6.0)) and (theta < (7 * np.pi / 6.0))):  # 限制与y轴夹角小于30度的线
            lines.append([rho, theta])
    lines.sort()
    lines_final = line_reduce(lines)
    return lines_final


def method_B(image,debug=False):
    
    #resize image
    r = 1024.0 / image.shape[1]
    dim = (1024, int(image.shape[0] * r))
 
    # perform the actual resizing of the image and show it
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    # print(image.shape)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edged = canny_edge(gray, 50, 150)
    proc_img, levels = connected_components(edged)

    proc_img = remove_short_clusters(proc_img, levels, th = 200)

    points = apply_hough_transform(proc_img, image, 130)
    
    lines, proc_img= merge_lines2(points, image)
    
    return lines, proc_img

def method_A(image):
    

    img_gray, img = pre_process(image)
    # cv2.imshow("pre",img_gray)
    
    edges = cv2.Canny(img, 50, 150)
    # cv2.imshow("image1",edges)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 160)
    # print(lines)
    lines1 = lines[:, 0, :]
    houghlines = line_sifting(lines1)
    img_show = draw(img_gray, houghlines)
    # print("pavel",len(houghlines))
    
    return houghlines,img_show





# --------------image segmentation---------------
def segmentation(img, lines,method):
    # print(lines)
    imgs = []
    i = 0
    j = 1
    img_height = 450
    if method=="B":
        img_height=lines[0][1]
    
    while i < len(lines) - 2:
        x1 = int(lines[i][0])
        x2 = int(lines[j][0])
        if(x1<0):
            print(x1)
            x1=0
        book_img = img[0:img_height, x1:x2]
        if(len(book_img)):
            # print(book_img)
            imgs.append(book_img)
        i = i + 1
        j = j + 1
    

    return imgs

def rotate_image(image):
    # cv2.imshow("original",image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edged = canny_edge(gray, 50, 150)
    proc_img, levels = connected_components(edged)

    proc_img = remove_short_clusters(proc_img, levels, th = 200)

    lines = cv2.HoughLines(proc_img.astype('uint8'), 1, np.pi/180, 130)
    
    actual_theta = []

    for values in lines:
        rho, theta = values[0]
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        t=0
        if(x2-x1 == 0):
            theta=90
        else:
            t= math.degrees(math.atan((y2 - y1) / (x2 - x1)))
        actual_theta.append(t)
        


    theta_values = {}
    for theta in actual_theta:
        theta = int(theta)
        if(theta in theta_values.keys()):
            theta_values[theta] += 1
        else:
            theta_values[theta] = 1

    # print(theta_values)
    dominant_dir = max(theta_values, key=theta_values.get)
    if(theta_values[dominant_dir]-theta_values[0]<5):
        dominant_dir=0
    # print(dominant_dir)
    r_angle=0
    if (dominant_dir< 0):
    
        r_angle = dominant_dir + 90
        
    elif(dominant_dir>0):
        
        r_angle = dominant_dir - 90

    fixed_image = imutils.rotate(image, r_angle)
    image=fixed_image
    # print(r_angle)
    # cv2.imshow("rotated",fixed_image)
    
    # cv2.waitKey()
    return fixed_image


def get_book_lines(img_path, debug = False):
    image = cv2.imread(img_path)
    image=cv2.resize(image,(720,960))
    image=rotate_image(image)
    # edged = canny_edge(gray, 50, 150, debug = debug)
    lines1,img1=method_A(image)
    lines2,img2=method_B(image)
    lines=[]

    method=""
    if(len(lines1)-len(lines2)>6 or len(lines2)-len(lines1)>50):
        image=img1
        lines=lines1
        method="A"
    else:
        image=img2
        lines=lines2
        method="B"

    
    img_segmentation = segmentation(image,lines,method)

    l=img_path.strip().split("/")
    str1=l[-1]
    print("-----------------Opening :"+str1+"--------------------")
    str1=str1[:-4]
    
    i = 0
    for img_s in img_segmentation:
        # cv2.imshow("error",img_s)
        if img_s.shape[0] == 0:
            print(i)
        # print(str1)
        string = os.path.join("./results",str1+"_"+str(i)+'.jpg')
        print("Write " + string)
        if not cv2.imwrite(string, img_s):
            raise Exception("Could not write image")
        
        i = i+1
    print("Results Succesfully Saved")
    # cv2.imshow("result",image)
    # cv2.waitKey()
    
    

    

# get_book_lines("./images/3.jpg")