import cv2


# Einlesen der Videodatei
cap = cv2.VideoCapture("center.mp4")
#cap = cv2.VideoCapture("highway.mp4")
#cap = cv2.VideoCapture(0)  # Webcam

# Object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)


while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    
    # Extract Region of interest
    roi = frame[340: 720,400: 800]
    
    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) # alles was nicht weiß ist, wird in der Maske ignoriert
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Leeres Array in dem Anzahl der erkannten Objekte gespeichert wird
    detections = []
    
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        # Makiere Objekt wenn es eine bestimmte Mindestgröße hat
        if area > 250:
            # EIne Box wird um Objekt gezeichnet
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            detections.append([x, y, w, h])
            
    # 2. Object Tracking
    print("Anzahl erkannter Objekte: ", len(detections))
    
    
    # Vorschaufenster
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    # Warte auf key event
    key = cv2.waitKey(30)
    if key == 27:
        break
        
cap.release()
cv2.destroyAllWindows()