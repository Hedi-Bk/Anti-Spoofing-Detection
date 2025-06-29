from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
import time


######################################>
# Parmas tha u eight change Based on your requirments 
classID = 0  # 0 is fake and 1 is real
offsetPercentageW = 10
offsetPercentageH = 20
confidence=90
floatingPoint = 6
blurThreshold =35
save =True
outputFolderPath = 'Dataset/DataCollect'
camWidth, camHeight = 640, 480
debug = False





######################################>

# Initialize the webcam
# '2' means the third camera connected to the computer, usually 0 refers to the built-in webcam
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

# Initialize the FaceDetector object
# minDetectionCon: Minimum detection confidence threshold
# modelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)
detector = FaceDetector()

    # Run the loop to continually get frames from the webcam
while True:
    # Read the current frame from the webcam
    # success: Boolean, whether the frame was successfully grabbed
    # img: the captured frame
    success, img = cap.read()
    imgOut = img.copy() #  faire d'autres traitements sans altérer img.

        # Detect faces in the image
        # img: Updated image
        # bboxs: List of bounding boxes around detected faces
    img, bboxs = detector.findFaces(img, draw=False) # 


    listBlur = []  # True False values indicating if the faces are blur or not
    listInfo = []  # The normalized values and the class name for the label txt file
        # Check if any face is detected
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:            # ---- Get Data  ---- #
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)

            if score > confidence :
                # ------  Adding an offset to the face Detected --------
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)


                # -------------- size.width>0 && size.height>0 -------------------
                if x<0 :x=0
                if y<0 :y=0


                #------------- Find Blurriness -------------------
                image_face = img[y:y+h, x:x+w ]


                gray= cv2.cvtColor(image_face,cv2.COLOR_BGR2GRAY)
                blurValue = int(cv2.Laplacian(gray, cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False) 

                 # ------  Normalize Values  --------
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2

                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)
                #print(xcn, ycn, wn, hn)


                # ------  To avoid values above 1 --------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1


                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                #------------- Drawing -------------------
                #cvzone.putTextRect(img, f'{blurvalue}', (x, y-10))
                cv2.rectangle(imgOut, (x, y,w, h),(0, 255, 0), 2)
                cvzone.putTextRect(imgOut, f'Score: {int(score)}% Blur: {blurValue}', (x, y - 10),scale=2, thickness=3)


                # ------  Drawing Normalized Values --------
                if debug :
                    cv2.rectangle(img, (x,y,w,h), (255, 0,0), 3) ## BGR
                    cvzone.putTextRect(img, f"score :{score} , Bluriness : {blurValue} " , (x-80, y-10),scale=2,thickness=3)
     

        if save:
            if all(listBlur) and len(listBlur)>0:
                # --- Save the image  ---- #
                timeNow =str(time.time()).split(".")[0] +str(time.time()).split(".")[1] 
                print(f'{str(time.time()).split(".")[0]} and  {str(time.time()).split(".")[1]}')
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
                # ------  Save Label Text File  --------
                for info in listInfo:
                        f=open(f"{outputFolderPath}/{timeNow}.txt", "a") 
                        f.write(info)
                        f.close()


    # Display the image in a window named 'Image'
    cv2.imshow("Image Dsiplay", imgOut)
    # Wait for 1 millisecond, and keep the window open
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
