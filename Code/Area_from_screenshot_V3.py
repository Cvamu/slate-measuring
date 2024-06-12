import cv2
import os
import numpy as np 
import math
import csv
import datetime

###############################################################################################

def convertImage1(image_path):
    xMin = 25
    xMax = -1
    yMin = 25
    yMax = -25

    # Read the image
    frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    frame = frame[yMin:yMax, xMin:xMax]

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Apply adaptive thresholding to segment the stones from the background
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=2)

    return morph

def convertImage2(image_path):
    xMin = 25
    xMax = -1
    yMin = 25
    yMax = -25

    contrast = 1   # 1 is no change
    brightness = 1 # 1 is no change 

    frame = cv2.imread(image_path)
    frame = frame[yMin:yMax, xMin:xMax]
    # Convert frame to greyscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(frame, 0, 100)
    grey = cv2.convertScaleAbs(grey, alpha=contrast, beta=brightness)
    
    # Threshold the greyscale image to separate the stones from the background
    ret, thresh = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY)

    return canny

def convertImage(image_path, greyscale_threshold):
    xMin = 25
    xMax = -1
    yMin = 25
    yMax = -25

    frame = cv2.imread(image_path)
    #frame = frame[yMin:yMax, xMin:xMax]

    #img = image.load_img(image_path, grayscale=True, target_size=(640, 480))
    #img = image.img_to_array(img, dtype='uint8')

    #greyscale_threshold = 140

    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _ , thresholded = cv2.threshold(grey_frame, greyscale_threshold, 255, cv2.THRESH_BINARY)
    #_ , thresholded2 = cv2.threshold(grey_frame, 230, 255, cv2.THRESH_BINARY)
    #thresholded = thresholded1 - thresholded2
    #thresholded = cv2.adaptiveThreshold(grey_frame, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,20)

    return thresholded


def isRectangular(corners, threshold_degrees=8):
    if len(corners) != 4:
        return False

    angles = []
    for i in range(4):
        # Calculate the angle between consecutive vertices
        pt1 = corners[i][0]
        pt2 = corners[(i + 1) % 4][0]
        pt3 = corners[(i + 2) % 4][0]

        vector1 = (pt1[0] - pt2[0], pt1[1] - pt2[1])
        vector2 = (pt3[0] - pt2[0], pt3[1] - pt2[1])

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        angle = math.acos(dot_product / (magnitude1 * magnitude2)) * (180 / math.pi)
        angles.append(angle)

    for angle in angles:
        if abs(90 - angle) > threshold_degrees:
            return False

    return True


def findArea(convertedImage):

    #scaleFactor = 30/7       # For 480p
    scaleFactor = 3.20
    min_stone_area = 50000   # Adjust this value as needed
    edge_margin = 5

    contours, _ = cv2.findContours(convertedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
            # Calculate area of the contour
            area = cv2.contourArea(contour) * scaleFactor**2

            # Check if the stone is above the minimum area threshold
            if area >= min_stone_area:
                # Approximate the contour to find the number of corners
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                num_corners = len(approx)

                # Check if the stone is far enough from the image edges
                x, y, w, h = cv2.boundingRect(contour)
                if (
                    #x >= edge_margin and
                    #y >= edge_margin and
                    x + w <= convertedImage.shape[1] - edge_margin and
                    y + h <= convertedImage.shape[0] - edge_margin and 
                    x + w >= 100 and
                    y + h >= 150 and
                    y < 350      and
                    convertedImage.shape[1]/2 > x and 
                    convertedImage.shape[1]/2 < x + w
                ):
                    if isRectangular(approx):
                        # Classify the shape as "rectangular"
                        shape = "rectangular"
                        side1_length = round(np.linalg.norm(approx[0] - approx[1])) * scaleFactor
                        side2_length = round(np.linalg.norm(approx[1] - approx[2])) * scaleFactor

                    else:
                        # Classify the shape as "irregular"
                        shape = "irregular"
                        side1_length, side2_length = 0, 0

                    return round(area), num_corners, shape, round(side1_length), round(side2_length), contour
    return 0


def imagePaths(folder):
    paths = []
    for filename in os.listdir(folder): 
        path = os.path.join(folder, filename)
        paths.append(path)
        paths.sort()

    return paths
    
###############################################################################################

def ProcessStone(path, csv_file, failed_folder):

    xMin = 25
    yMin = 25

    image = cv2.imread(path)
    image2 = cv2.imread(path)
    #cv2.imshow('Image', image)
    #cv2.imshow('Canny', convertedImage)

    date = path[21:31]
    pathtime = path[38:46]
    timestamp = datetime.datetime.strptime(pathtime, "%H-%M-%S").time()

    greyscale_threshold = 140
    run = True
    while run:
        
        convertedImage = convertImage(path, greyscale_threshold)
        canny = convertImage2(path)
        #print(greyscale_threshold)
        try:
            area, num_corners, shape, side1_length, side2_length, contour = findArea(convertedImage)
            #print(f"Area of stones: {area} square mm")
            #cv2.drawContours(image, contour, -1, (0, 255, 0), 2, offset=(xMin,yMin))
            #cv2.line(image, (xMin+100, 0), (xMin+100, 360), (0, 255, 0), 2) 
            #cv2.putText(image, f"Area, stone: {area}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #cv2.putText(image, f"Threshold value: {greyscale_threshold}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #cv2.imshow('Image', image)
            
            stone = [pathtime, area, num_corners, shape, side1_length, side2_length]
            with open(csv_file, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(stone)
            #stone_index += 1
            #totalArea += area / 1000000
                
            cv2.imshow('BlackAndWhite', convertedImage)
            cv2.imshow('Image', image)
            #cv2.imshow('Image2', image2)
            #cv2.imshow('Canny', canny)
            #cv2.moveWindow('Image', 300, 50)  
            #cv2.moveWindow('BlackAndWhite', 300, 550)  
            cv2.waitKey(1)

            return area/1000000, shape, timestamp, image

            run = False

        except Exception as e:
            print(f"An error occurred: {e} for stone {path}")
            greyscale_threshold += 5

        #finally:
        #    greyscale_threshold += 5
            

        if greyscale_threshold > 180: 
        #print(f"An error occurred: {e} for stone {path}")
            stone_path = os.path.join(failed_folder, f"stone_{date}_{pathtime}.png")
            cv2.imwrite(stone_path, image)
            cv2.putText(image, f"Failed, stone {path}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #cv2.imshow('Image', image)
            #cv2.imshow('BlackAndWhite', convertedImage)
            run = False
            return None, None, None, None
        
        
###############################################################################################

folder = "AreaData/screenshots_2024-05-08"
csv_file = "results_day1.csv"
failed_folder = "failed"
os.makedirs(failed_folder, exist_ok=True)

csv_header = ['Timestamp', 'Area', 'Corners', 'Shape', 'Side1_Length', 'Side2_Length']
with open(csv_file, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(csv_header)

paths = imagePaths(folder)
paths = [None, "AreaData/screenshots_2024-05-08/stone_09-07-42.png"]
print(paths)

for path in paths[1:]:
    ProcessStone(path, csv_file, failed_folder)

    if cv2.waitKey(0) & 0xFF == ord('q'):  break