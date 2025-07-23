# Object Tracking using YOLO and OpenCV for CCTV surveillance

This project implements ** real-time person tracking** using **YOLO11** for detection and **StrongSORT** & **ByteTrack** for tracking. It includes features like:
- ROI (Region of interest) based tracking.
- Duration logging per ROI.
- ROI configuration and results export in pyyaml.
- CSV export for entry/exit times with respect to ROIs.

---

## **System Environment**

This project has been tested under the following setup:
- **OS:** Name: Windows 11, Version: 10.0.26100
- **Pthon:** 3.13.2
- **IDE:** Pycharm 2024.2
- **YOLO11:** ultralytics 8.3.149
- **OpenCV-python:** 4.11.0
- **Shapely:** 2.1.1
- **Numpy:** 2.2.4
- **torch:** 2.7.0
- **torchvision:** 0.22.0

 ---

 ## **Prerequisities**

 Before running this project, install the required dependencies:
 ### **Install Python Packages**
In PyCharm Terminal:
'''bash
pip installl ultralytics opencv-python shapely numpy pyyaml torch torchvision lap

---

## How to configure

1. clone the repository: https://github.com/Vaibhav-Dhengle/Object-Tracking.git
2. Create and active the virtual environment:
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



 
