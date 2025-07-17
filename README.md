## Smart-Traffic-Management-System for Urban Congestion Reduction

<div align="justify">
  
### Overview

<video src="fundamental_problem.mp4" width="700" controls></video>

> The video outlines the traffic congestion issues addressed by our proposed system.

Urban environments today are grappling with ever-increasing vehicular traffic, leading to frequent congestion, extended waiting times at intersections, and inefficient fuel consumption. Conventional traffic light systems operate on fixed cycles, lacking the adaptability to respond to real-time road conditions or emergency situations. This not only disrupts traffic flow but also hinders timely passage of critical services like ambulances and fire brigades. Addressing this pressing urban mobility challenge demands an intelligent and adaptive approach to traffic control.

Our solution proposes a smart, IoT-enabled traffic control system that leverages real-time computer vision through **YOLOv8** for vehicle detection and dynamically adjusts traffic light signals based on road congestion levels and emergency vehicle detection. The system includes a prediction layer that analyzes traffic patterns using machine learning, providing actionable insights to improve flow and reduce congestion.

### Key Features ✨
- **Real-time Vehicle Detection** using YOLOv8 (Ultralytics) to analyze traffic density and identify multiple vehicle types.
- **Traffic Flow Prediction** using machine learning models with automated evaluation (MAE, RMSE, R²).
- **Interactive Dashboard** built with Streamlit for real-time monitoring, visual insights, and user interaction.
- **Modular Codebase** with clearly separated components for detection, prediction, and visualization.

### Getting Started
#### Step 1: Clone the Repository
```bash
git clone https://github.com/Smarth2005/Smart-Traffic-Management-System.git
cd Smart-Traffic-Management-System
```

#### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

#### Step 3: Launch the Streamlit Dashboard from the terminal
```python
streamlit run streamlit_app.py
```

#### Step 4: Run Vehicle Detection Pipeline
```python
python app2.py
```
>`app2.py` serves as the main inference pipeline for YOLOv8-based vehicle detection and interacts with video frames in real-time or from stored samples.

**Vehicle Detection Demonstration**: Due to file size constraints, video samples have not been included in this repository. [Download Test Videos](https://drive.google.com/drive/u/0/folders/1wbxnLHmrt0wVk3cB9Hkf3qOWASlVfsEI)

### Future Enhancements:-
- Deployment-ready backend with cloud or edge computing.
- Dynamic Traffic Signal Timing based on real-time congestion and emergency vehicle presence.
- Simulated junction visualization and real-time signal strategy comparison via SUMO-GUI.

### Contributors: 
This project was undertaken as part of an academic internship at the **Experiential Learning Centre (ELC), Thapar Institute of Engineering and Technology, Patiala**, under the esteemed mentorship of **Dr. Amit Kumar Trivedi**, Professor, Department of Computer Science. 

**Team Members:** [Anchit Mehra](https://github.com/AnMaster15)  | [Ayush Sharma] | [Samdeep Sharma](https://github.com/SamdeepSharma) | [Smarth Kaushal](https://github.com/Smarth2005) | [Aniket Gupta](https://github.com/AniketGupta27) | [Pavni Goel]

>*We believe this project lays the groundwork for smarter, AI-powered urban mobility solutions. Your suggestions, feedback, and contributions are most welcome as we continue building responsive and intelligent traffic systems for the real world.*

### License
This project is distributed under the terms of the MIT License. See `LICENSE` for more details.

### Contributing
We welcome contributions that help improve this project! If you have ideas for new features, suggestions for enhancements, or have found a bug, feel free to open an issue or submit a pull request.
Follow these steps to contribute:
1. Fork the repository.
2. Clone your fork to your local machine.
3. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature-name
   ```
4. Commit your changes and push:
   ```bash
   git commit -m "Add new feature"
   git push origin feature-branch
   ```
5. Create a pull request from your branch.

Happy Coding!
</div>
