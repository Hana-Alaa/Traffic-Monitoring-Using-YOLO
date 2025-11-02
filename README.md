#  Traffic Monitoring Using YOLO

A simple real-time project that detects, tracks, and counts vehicles using **YOLOv8**.  
It identifies different vehicle types (cars, buses, trucks, motorcycles, etc.), counts how many pass in each direction, and estimates their speeds from video footage.  

---

## Features  
- Vehicle detection using **YOLOv8x** pretrained model  
- Real-time **tracking** with ByteTrack  
- **Speed estimation** based on object movement  
- **Counting** vehicles moving up and down  
- Supports video input and saves annotated output  

---

## Requirements  
Install the dependencies:  
```bash
pip install ultralytics torch opencv-python
```
---

## 📂 Project Structure
```
Traffic-Monitoring-using-YOLO/
│
├── vehicle_detection.py      # Main script (your code)
├── videos/                   # Input videos
│   └── test1.mp4
├── vehicle_count_speed.mp4   # Output video
├── README.md                 # Project documentation
└── requirements.txt          # Optional: list of dependencies
```

---

## Usage
1. Place your input video inside the videos/ folder.
2. Run the script:
```
python vehicle_detection.py
```
3. The processed video will be saved as:
```
vehicle_count_speed.mp4
```

---

## Output Example

The output video displays:
- Bounding boxes around vehicles
- Object ID, type, and estimated speed (in km/h)
- Count of vehicles moving up and down
- 
---

## Model
- YOLOv8x from Ultralytics
- Tracker: ByteTrack


