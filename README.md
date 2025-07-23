# Object Tracking using YOLO and OpenCV for CCTV surveillance

This project implements ** real-time person tracking** using **YOLO11** for detection and **StrongSORT** & **ByteTrack** for tracking. It includes features like:
- ROI (Region of interest) based tracking.
- Duration logging per ROI.
- ROI configuration and results export in pyyaml.
- CSV export for entry/exit times with respect to ROIs.

---

## **System Environment**

This project has been tested under the following setup:
- **OS:** Windows 11
- **Pthon:** 3.10.12
- **IDE:** Pycharm 2024.2
- **YOLO11:** ultralytics
- **OpenCV:** 4.10+
- **Shapely:** 2.0+

 ---

 ## **Prerequisities**

 Before running this project, install the required dependencies:
 ### **Install Python Packages**
In PyCharm Terminal:
pip installl ultralytics opencv-python shapely numpy pyyaml torch torchvision lap

---

## How to configure

1. clone the repository
2. Create virtual environment
In Terminal give following commands:
1] python -m venv .venv
2] .venv\Scripts\activation

---

## **CSV Export**
- Tracking logs are automatically exported to tracking_summary_botsort.csv or tracking_summary_bytetrack.csv
- Columna include Person ID, ROI Entry Time, ROI Exit Time, etc

---

## **Future Improvements**
- Advanced appearance-based Re_ID for stronger occlusion handling.
- Web-based dashboard for real-time tracking visualization.



 
