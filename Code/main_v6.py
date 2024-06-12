import threading
import schedule
from queue import Queue

import datetime 
import time

import cv2
import os
import serial
import keyboard
import csv

from Area_from_screenshot_V3 import ProcessStone

import tkinter as tk
from tkinter import ttk

###############################################################################################
### Capture Area ###

def isSignificantNonBlack(frame, area_threshold=0.05, greyscale_threshold=100):
    # Make greyscale image
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the greyscale frame to identify non-black pixels
    _ , thresholded = cv2.threshold(grey_frame, greyscale_threshold, 255, cv2.THRESH_BINARY)
    #thresholded = cv2.adaptiveThreshold(grey_frame,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

    # Calculate the percentage of non-black pixels
    non_black_percentage = cv2.countNonZero(thresholded) / (frame.shape[0] * frame.shape[1])

    return non_black_percentage > area_threshold


def stoneInCenter(frame, contour):

    scaleFactor = 3.19       # For 480p
    #scaleFactor = 10/7       # For 1080p
    min_stone_area = 25000   # Adjust this value as needed
    edge_margin = 10

    #for contour in contours:
    # Calculate area of the contour
    area = cv2.contourArea(contour) * scaleFactor**2

    # Check if the stone is above the minimum area threshold
    if area >= min_stone_area:

        # Check if the stone is far enough from the image edges
        x, y, w, h = cv2.boundingRect(contour)
        if (x >= edge_margin) and (x + w <= frame.shape[1] - edge_margin):
            #print("Area:", area, ". Edge margin top:", y, ". Edge margin bottom:", frame.shape[0] - (y+h))

            top_margin = x
            bottom_margin = frame.shape[1] - (x+w)

            total_margin = top_margin + bottom_margin

            if abs(top_margin - total_margin/2) < 5:
                #print(area)
                return True
            
    return False


def captureStones(video_path, endTime):

    endTime = datetime.datetime.strptime(endTime, "%H:%M:%S").time()

    # Initiate video and output folder
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)    # 0
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 100) # 100
    cap.set(cv2.CAP_PROP_CONTRAST, 200)   # 500
    #cap.set(cv2.CAP_PROP_SETTINGS, 0)     # Restore default settings for camera
    
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    AreaFolder = "AreaData"
    output_folder = f"screenshots_{date}"
    output_path = os.path.join(AreaFolder, output_folder)
    os.makedirs(output_path, exist_ok=True)
    

# Initiate values
    frame_count = 0
    stone_count = 0

    xMin =    0
    xMax =   -1
    yMin =  135
    yMax = -165

    greyscale_threshold = 140 # 130

    frame_count_last = 0

    print("Video ready")

# Read video feed
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        fullFrame = frame.copy()
        frame = frame[yMin:yMax, xMin:xMax]

        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _ , thresholded = cv2.threshold(grey_frame, greyscale_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow('Live Feed', thresholded)

    # Check conditions
        if isSignificantNonBlack(frame, greyscale_threshold=greyscale_threshold):
            for contour in contours:
                ifStone  = stoneInCenter(frame, contour)
                if ifStone:
                    frame_count_now = frame_count
                    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
                    if abs(frame_count_now - frame_count_last) > 30:
                        stone_count += 1
                        path = f"stone_{timestamp}.png"
                        stone_path = os.path.join(output_path, path)
                        #stone_path = output_folder + "/" + path
                        #stone_path = os.path.join(AreaFolder, f"screenshots_{date}", f"stone_{timestamp}.png")
                        #print(stone_path)

                        #cv2.drawContours(fullFrame, contour, -1, (0, 255, 0), 2, offset=(xMin,yMin))
                        cv2.imwrite(stone_path, fullFrame)
                        frame_count_last = frame_count_now

                        path_queue.put(stone_path)

        frame_count += 1
        
        current_timestamp = datetime.datetime.now().time()
        if cv2.waitKey(1) & 0xFF == ord('q'):  break
        if current_timestamp > endTime: break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames and captured {stone_count} grey stones.")

###############################################################################################
### Process Image ###
       
def ProcesingOfImages(endTime):

    endTime = datetime.datetime.strptime(endTime, "%H:%M:%S").time()
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    csv_file = f"data_area_{date}.csv" 

    failed_folder = "Failed"
    os.makedirs(failed_folder, exist_ok=True)

    csv_header = ['Timestamp', 'Area', 'Corners', 'Shape', 'Side1_Length', 'Side2_Length']
    with open(csv_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_header)

    print("Processing ready")

    while True:
        path = path_queue.get()

        if path is None: 
            path_queue.task_done()
            break
        else:
            print(path)
            area, shape, timestamp, image = ProcessStone(path, csv_file, failed_folder)
            path_queue.task_done()

            if area != None: 
                #product = SortStonesAreaOnly(area, shape)
                stone = {"area": area, "shape": shape, "timestamp": timestamp}
                area_queue.put(stone)
                #cv2.imshow('Image', image)
                #cv2.moveWindow('Image', 300, 250)  
                #cv2.waitKey(1)
        
        current_timestamp = datetime.datetime.now().time()
        if current_timestamp > endTime: break
    
    print("Processing ended")
            

###############################################################################################
### Get Height ###

def ultrasonicToThickness(COMport1, COMport2, endTime):

    arduino1 = serial.Serial(port=COMport1, baudrate=9600)
    arduino2 = serial.Serial(port=COMport2, baudrate=9600)

    endTime = datetime.datetime.strptime(endTime, "%H:%M:%S").time()
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    csv_file = f"HeightData/heights_{date}.csv"

    zeroLevel1, zeroLevel2 = 0, 0
    zMax1, zMax2 = 1660, 1675

    windowSize = 10
    dataStream1 = [0 for i in range(windowSize)]
    dataStream2 = [0 for i in range(windowSize)]
    list1 = []
    list2 = []
    

    csv_header = ['height', 'timestamp']
    with open(csv_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_header)

    print("IR laser distance ready")

    while True:
        try:
            data1 = float(arduino1.readline())
            data2 = float(arduino2.readline())
        except:
            data1, data2 = zMax1, zMax2
            print("ups")
        dataStream1.pop(0)
        dataStream2.pop(0)
        dataStream1.append(data1)
        dataStream2.append(data2)
        
        if (zeroLevel1 == 0) and (0 not in dataStream1): zeroLevel1 = data1   
        if (zeroLevel2 == 0) and (0 not in dataStream2): zeroLevel2 = data2  
        if zeroLevel1-data1 < 5: zeroLevel1 = data1
        if zeroLevel2-data2 < 5: zeroLevel2 = data2
        if zeroLevel1 > zMax1: zeroLevel1 = zMax1
        if zeroLevel2 > zMax2: zeroLevel2 = zMax2

        height1 = zeroLevel1 - data1
        height2 = zeroLevel2 - data2
        list1.append(height1)
        list2.append(height2)

        height = max(height1, height2)

        current_timestamp = datetime.datetime.now().time()
        """
        if (height1 > 8) and (height1 < 100):
             with open(csv_file, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([height1, current_timestamp])
                height_data = {'timestamp': current_timestamp, 'height': height1}
                height_queue.put(height_data)
        if (height2 > 8) and (height2 < 100):
             with open(csv_file, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([height2, current_timestamp])
                height_data = {'timestamp': current_timestamp, 'height': height2}
                height_queue.put(height_data)
        """
        if (height > 8) and (height < 100):
             with open(csv_file, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([height, current_timestamp])
                height_data = {'timestamp': current_timestamp, 'height': height}
                height_queue.put(height_data)

        while len(list1) >= 10: list1.pop(0)
        while len(list2) >= 10: list2.pop(0)

        if 0 not in list1: zeroLevel1 = data1
        if 0 not in list2: zeroLevel2 = data2

        if keyboard.is_pressed('q'):    break
        if current_timestamp > endTime: break

    print("IR laser ended")

###############################################################################################
### Pair Area and Height Data ###

def pairHeightAndArea(endTime):

    endTime = datetime.datetime.strptime(endTime, "%H:%M:%S").time()

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    result_file = f"data_{date}.csv"

    csv_header = ['timestamp','area','shape','height']
    with open(result_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_header)

    heightBuffer = []
    areaBuffer   = []
    maxTimedelta = datetime.timedelta(seconds= 2)
    minTimedelta = datetime.timedelta(seconds= 2) # Husk minus fortegn senere
    bufferTime   = datetime.timedelta(seconds= 5)

    print("Pairing ready")

    while True:
            current_timestamp = datetime.datetime.now().time()
            cuttoffTime = (datetime.datetime.combine(datetime.date(1,1,1), current_timestamp) - bufferTime).time()

            while not height_queue.empty():
                heightBuffer.append(height_queue.get())
                height_queue.task_done()
            while not area_queue.empty():
                areaBuffer.append(area_queue.get())
                area_queue.task_done()

            if len(heightBuffer) > 0: 
                if heightBuffer[0]["timestamp"] < cuttoffTime: heightBuffer.pop(0)
            if len(  areaBuffer) > 0:
                if   areaBuffer[0]["timestamp"] < cuttoffTime:   

                    stone = [areaBuffer[0]["timestamp"], areaBuffer[0]["area"], areaBuffer[0]["shape"], 0]
                    with open(result_file, 'a', newline='') as f:
                            csv_writer = csv.writer(f)
                            csv_writer.writerow(stone)

                    stone_data = {"area":areaBuffer[0]["area"], "height":0, "shape":areaBuffer[0]["shape"], "timestamp":areaBuffer[0]["timestamp"]}
                    stone_queue.put(stone_data)
                    
                    areaBuffer.pop(0)

            if (len(heightBuffer) > 0) and (len(areaBuffer)> 0):
                
                flag = False
                for a in range(len(areaBuffer)): 
                    area  = areaBuffer[a]["area"]
                    shape = areaBuffer[a]["shape"]
                    area_timestamp = areaBuffer[a]["timestamp"]
                    for h in range(len(heightBuffer)):
                        height = heightBuffer[-h]["height"]
                        height_timestamp = heightBuffer[h]["timestamp"]

                        timestamp_h = datetime.datetime.combine(datetime.date(1,1,1), height_timestamp)
                        timestamp_a = datetime.datetime.combine(datetime.date(1,1,1), area_timestamp)
                        #difference = (datetime.datetime.combine(datetime.date(1,1,1), height_timestamp) - area_timestamp).time()
                        #difference = datetime.datetime.combine(datetime.date.today(), height_timestamp) - datetime.datetime.combine(datetime.date.today(), area_timestamp)
                        #if (difference > minTimedelta) and (difference < maxTimedelta):
                        #if ((area_timestamp - minTimedelta) <= height_timestamp <= (area_timestamp + maxTimedelta)):
                        if ((timestamp_a - minTimedelta) <= timestamp_h <= (timestamp_a + maxTimedelta)):
                            
                            stone = [area_timestamp, area, shape, height]

                            with open(result_file, 'a', newline='') as f:
                                    csv_writer = csv.writer(f)
                                    csv_writer.writerow(stone)
                            
                            stone_data = {"area": area, "height":height, "shape": shape, "timestamp": area_timestamp}
                            stone_queue.put(stone_data)

                            areaBuffer.pop(a)
                            heightBuffer.pop(-h)
                            flag = True
                            break
                    if flag == True: break
            
            current_timestamp = datetime.datetime.now().time()
            if current_timestamp > endTime: break
            
            time.sleep(1)

    print("Pairing ended")


###############################################################################################
### Sort Stones ###
            
def SortStonesAreaOnly(area, shape):
    if shape == "irregular":
        if   (area >= 0.100) and (area < 0.200): product = "s"
        elif (area >= 0.200) and (area < 0.500): product = "m"
        elif (area >= 0.500) and (area < 1.200): product = "l"
        else:                                    product = "irr"
    elif shape == "rectangular":                 product = "rec"

    return product

def sortStone(area, shape, height):
    if shape == "irregular":
        if (area >= 0.100) and (area < 0.200):
            if   (height >= 10) and (height < 20): product = "s1020"
            elif (height >= 20) and (height < 30): product = "s2030"
            elif (height >= 30) and (height < 40): product = "s3040"
            else:                                  product = "s"
        elif (area >= 0.200) and (area < 0.500):
            if   (height >= 10) and (height < 20): product = "m1020"
            elif (height >= 20) and (height < 30): product = "m2030"
            elif (height >= 30) and (height < 40): product = "m3040"
            elif (height >= 40) and (height < 60): product = "m4060"
            else:                                  product = "m"
        elif (area >= 0.500) and (area < 1.200):
            if   (height >= 10) and (height < 20): product = "l1020"
            elif (height >= 20) and (height < 30): product = "l2030"
            elif (height >= 30) and (height < 40): product = "l3040"
            else:                                  product = "l"
        else:                                      product = "irr"
    elif shape == "rectangular": product = "rec"

    return product

def sort(endTime):

    endTime = datetime.datetime.strptime(endTime, "%H:%M:%S").time()

    print("Sorting ready")

    while True:
        stone = stone_queue.get()

        if stone is None: 
            stone_queue.task_done()
            break
        else:
            area  = stone["area"]
            height = stone["height"]
            shape  = stone["shape"]
            product = sortStone(area, shape, height)

            prod_data = {"product":product, "area":area}
            product_queue.put(prod_data)

            stone_queue.task_done()
        
        current_timestamp = datetime.datetime.now().time()
        if current_timestamp > endTime: break
    
    print("Sorting ended")



###############################################################################################
### GUI ###
       
class SlateGUI:
    def __init__(self, master):
        self.master = master
        master.title("Slate Packing Quantities")

        self.categories = ['s1020', 's2030', 's3040', 's', 
                           'm1020', 'm2030', 'm3040', 'm4060', 'm', 
                           'l1020', 'l2030', 'l3040', 'l', 
                           'irr', 'rec']
        self.quantities = {category: 0 for category in self.categories}  # Simulated quantities
        self.pallets_packed = {category: 0 for category in self.categories}
        self.category_components = {}  # Corrected approach to store GUI components

        for i, category in enumerate(self.categories):
            label_text = f"{category}: 0.00 sqm, Pallets Packed: 0"
            label = ttk.Label(master, text=label_text)
            label.grid(row=i, column=0, sticky=tk.W)

            reset_button = ttk.Button(master, text="Reset Pallet", command=lambda c=category: self.reset_pallet(c))
            reset_button.grid(row=i, column=1, padx=5)

            # Correctly storing the components in a dictionary
            self.category_components[category] = {"label": label, "reset_button": reset_button}

        self.total_amount_label = ttk.Label(master, text="Total on Current Pallets: 0.00 sqm", background="lightgrey")
        self.total_amount_label.grid(row=len(self.categories), column=0, columnspan=2, sticky=tk.W, pady=4)

        self.update_quantities()

    def update_quantities(self):
        self.simulate_incoming_slate()
        total_sqm = sum(self.quantities.values())
        for category, components in self.category_components.items():
            quantity = round(self.quantities[category], 2)
            label_text = f"{category}: {quantity} sqm, Pallets Packed: {self.pallets_packed[category]}"
            components["label"].config(text=label_text)

            # Ensure the label's background changes to red if quantity exceeds 15, else keep it lightgrey
            if quantity >= 15:
                components["label"].config(background="red")
            else:
                components["label"].config(background="SystemButtonFace")  # Use default system background

        self.total_amount_label.config(text=f"Total on Current Pallet: {round(total_sqm, 2)} sqm")
        self.master.after(1000, self.update_quantities)  # Schedule next update

    def reset_pallet(self, category):
        self.quantities[category] = 0  # Reset the quantity for the specific category
        self.pallets_packed[category] += 1  # Increment the pallet counter for the category
        self.update_quantities()  # Immediately refresh the display

    def simulate_incoming_slate (self):
        if not product_queue.empty():
            stone = product_queue.get()
            area = stone["area"]
            category = stone["product"]
            self.quantities[category] += area
            product_queue.task_done()
            #self.update_display()
    
def GUI():
    root = tk.Tk()
    #gui = ProductPalletGUI(root)
    gui = SlateGUI(root)
    print("GUI ready")
    root.mainloop()


###############################################################################################
### Define Scripts ###
        
def run_script1(video_path, endTime):
    captureStones(video_path, endTime)

def run_script2(COMport1, COMport2, endTime):
    ultrasonicToThickness(COMport1, COMport2, endTime)

def run_script3(endTime):
    ProcesingOfImages(endTime)

def run_script4(endTime):
    pairHeightAndArea(endTime)

def run_script5(endTime):
    sort(endTime)

def run_script6():
    GUI()

###############################################################################################
### Start Threads ###

def main_script(COMport1, COMport2, video_path, endTime):
    script1_thread = threading.Thread(target=run_script1, args=( video_path, endTime ))
    script2_thread = threading.Thread(target=run_script2, args=( COMport1, COMport2, endTime ))
    script3_thread = threading.Thread(target=run_script3, args=(             endTime,))
    script4_thread = threading.Thread(target=run_script4, args=(             endTime,))
    script5_thread = threading.Thread(target=run_script5, args=(             endTime,))
    script6_thread = threading.Thread(target=run_script6)

    script1_thread.start()
    script2_thread.start()
    script3_thread.start()
    script4_thread.start()
    script5_thread.start()
    script6_thread.start()
   
    script1_thread.join()
    script2_thread.join()
    script3_thread.join()
    script4_thread.join()
    script5_thread.join()
    script6_thread.join()


###############################################################################################
### Use Schedule ###
    
def main(COMport1, COMport2, video_path, startTime, endTime): 
    schedule.every().monday.at(   startTime).do(main_script, COMport1, COMport2, video_path, endTime)
    schedule.every().tuesday.at(  startTime).do(main_script, COMport1, COMport2, video_path, endTime)
    schedule.every().wednesday.at(startTime).do(main_script, COMport1, COMport2, video_path, endTime)
    schedule.every().thursday.at( startTime).do(main_script, COMport1, COMport2, video_path, endTime)
    schedule.every().friday.at   (startTime).do(main_script, COMport1, COMport2, video_path, endTime)

    while True:
        schedule.run_pending()

###############################################################################################
### Run Main ###
        
path_queue    = Queue()
height_queue  = Queue()  
area_queue    = Queue()
stone_queue   = Queue()
product_queue = Queue()

        
if __name__ == "__main__":
    #main("COM3", "COM4", 0, "06:55:00", "15:05:00")
    main_script("COM3", "COM4", 0, "18:05:00")
    