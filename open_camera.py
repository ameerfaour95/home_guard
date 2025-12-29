import cv2
import os

# 1. Allow glitches (don't discard corrupt frames)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# 2. Use the Sub Stream (s1)
RTSP_URL = "rtsp://admin:amer1967%40@192.168.68.110:554/unicast/c6/s1/live"

print(f"Connecting to: {RTSP_URL}")
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("FATAL: Cannot connect. Check IP/Network.")
    exit()

print("Connected! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Frame drop...")
        continue

    # Show the video
    cv2.imshow("Test Stream", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()