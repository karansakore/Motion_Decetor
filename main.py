import cv2, time, pandas
from datetime import datetime

first_frame = None
status_list = [None, None]
times = []
# DataFrame to store the time values during which object detection and movement appears
df = pandas.DataFrame(columns=["Start", "End"])

# Creating a video capture object to record video using web cam
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status = 0  # Status at the begining of the recording is zero as the object is not visible
    # Convert the frame color to the gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the gray scale frame to GaussianBlur
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # This if statement used to store the first frame/image of the video
    if first_frame is None:
        first_frame = gray
        continue

    # Calculate difference between first frame and other frames
    delta_frame = cv2.absdiff(first_frame, gray)

    # Provides a threshold value, such that it will convert the difference value with less than 30 to black.
    # if the difference is greater than 30 it will convert those pixels to white.
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=0)

    # Define the counter area. Basically, add the borders
    cnts, hierachy = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        # Remove noices and shadows. Basically, it will keep only that part white, which has area greater than 1000
        # pixels.
        if cv2.contourArea(contour) < 1000:
            continue
        status = 1  # Change in status when the object is being detected
        # Creates a rectangle box around the object in the frame
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    status_list.append(status)  # List of status for every frame

    status_list = status_list[-2:]

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow('frame', frame)
    cv2.imshow('Capturing', gray)
    cv2.imshow('delta', delta_frame)
    cv2.imshow('thresh', thresh_delta)

    key = cv2.waitKey(1)  # Frame will change in 1 millisecond
    if key == ord('q'):  # This will break the loop, once the user press 'q'.
        break

print(status_list)
print(times)

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i + 1]}, ignore_index=True)  # store the values in a DataFrame

df.to_csv("Times.csv")  # Write the DataFrame to a CSV file

video.release()
cv2.destroyAllWindows()
