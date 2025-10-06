# Realtime_pothole_detection
This project leverages computer vision and AI to detect potholes and road damage in real time using cameras installed on government vehicles. The system tracks, categorizes, and estimates repair costs for each detected pothole, helping streamline road maintenance efforts.


## Features
* Pothole and Damage Detection: <br>
Cameras mounted on government vehicles (e.g., trash collectors, patrol cars) capture video footage of roads. AI models analyze this footage to detect potholes and road damage in real time.
* Pothole Count Between Locations: <br>
The system counts the number of potholes detected between two specific locations (e.g., Point A to Point B), assisting in prioritizing repair efforts on critical routes.
* Location and Database Management:<br>
Each detected pothole is tagged with precise GPS coordinates and stored in a central database. The system logs the time and date of detection for future reference and easy querying.


## Technologies Used
* Computer Vision & AI for real-time pothole detection.
* GPS Integration for accurate geolocation tagging.
* Database for storing and managing pothole detection data.

## Usage
* For Detection: The system continuously analyzes video from cameras mounted on vehicles.
* For Reporting: Detected potholes are automatically uploaded to the web interface with GPS locations.

## Future enhancements
* Automated Reporting to Authorities: Detected damage is reported to the government road department via a web interface. Employees can view the data sorted by date and time, with access to interactive maps showing the damaged locations.
* Real-time Alerts: Improve the notification system to include push notifications for mobile devices.
* Advanced Analytics: Introduce advanced analytics and machine learning for better road condition predictions.
* Mobile App: Develop a mobile app for employees to manage repairs directly from the field.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Requirements 
Refer to requirements.txt to know the versions required to run the application<br>
### `pip install -r requirements.txt` <br>
is the command to install Python packages

## Demo
1) To run the code, go to the terminal and enter command -
   ### `streamlit run home.py`
   After running, the below screen is displayed
   ![image](https://github.com/user-attachments/assets/8a449dfd-c7cf-4e52-9697-52445ace322a)

2) There are 3 options available for the user to choose from
* Image Detection
* Video Detection
* Real-time Detection

## Image Detection
![image](https://github.com/user-attachments/assets/7ea61047-f840-4f6c-bd03-b95a4e555098)
The user can now upload a picture that is present on the device from which the potholes will be identified

![image](https://github.com/user-attachments/assets/bf1041cf-1b68-4048-aa78-a0ba099fc355)
The Output can also be downloaded

## Video Detection 
![image](https://github.com/user-attachments/assets/071d2389-a91c-434a-a9d1-4794d23e06d0)
A similar drag-and-drop feature is available for uploading videos to identify potholes. The maximum size of the video is 1GB.

![image](https://github.com/user-attachments/assets/60c707ff-2780-4b72-b135-552d196e5f2b)
The video is processed bit by bit and the potholes are identified

## Real-time Detection 
Detect the road damage in real time using a USB Webcam. This can be useful for on-site monitoring with personnel on the ground. Select the video input device and start the inference.
When a pothole is detected both the longitudes and the latitudes (the Geocoordinates of the device) are identified by an API call and stored in a database.
There is an option to select a device with a video feed available.
![image](https://github.com/user-attachments/assets/072d8fdf-e86a-4942-9165-d9c22fd22522)
