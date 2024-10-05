# PosePerfect
PosePerfect is a Python project that allows you to edit the pose of an object within a scene using cutting-edge generative AI techniques. This project focuses on two key tasks:

## Object Segmentation: 
### PosePerfectT1.ipynb (.ipynb for quicker view of the code and the output), PosePerfectTask1 (to run the code)

What it does: Isolates the target object from the background image.
How it works: Takes an input image and a text prompt specifying the object class (e.g., chair, table).
Output: A segmented image with a red mask highlighting the object's boundaries.
This code utilizes YOLOv5 for object detection and the Segment Anything Model (SAM) for segmentation to detect, segment, and visually highlight specified objects in images.

### Key Components
#### Library Imports:

OpenCV (cv2): For image processing (reading, writing).

NumPy (np): For numerical operations on image data.

Torch: To load and run the YOLOv5 and SAM models.

Matplotlib (plt): For displaying images.

#### Configuration:

The model path (CHECKPOINT_PATH) and type (MODEL_TYPE) are set, and the device is configured to use a GPU if available.

#### Models:

SAM is loaded using its checkpoint and set to the specified device.
An instance of SamAutomaticMaskGenerator is created for segmentation.
The YOLOv5 model is loaded from Ultralytics’ repository as a pre-trained model.

#### Image Display Function:

The display_image function reads an image, converts it from BGR to RGB, and displays it using Matplotlib.

#### Detection and Segmentation Function:

The function detects objects in the image and checks for the target object.
If detected, it extracts the region of interest (ROI), segments it using SAM, and applies a red highlight before saving and displaying the final image.

![chair](https://github.com/user-attachments/assets/ec8ac2c6-1ce0-4379-8c3d-1a7f555ea1be)
![highlighted_chair](https://github.com/user-attachments/assets/d8aef62b-5aa1-46ec-9c1f-ed89588a32dc)

## SetUp
1. Open VSCode 
2. Copy the given command
```shell
git clone https://github.com/saarinii/PosePerfect.git
```
3. Navigate to the project directory:
```shell
cd PosePerfectTask1
```
4. Install the required packages:
```shell
pip install -r requirements.txt
```
5. You can now run the script as follows::
```shell
python run.py --image ./example.jpg --class "chair" --output ./generated.png
```
Replace ./example.jpg with the path to your input image, "chair" with the object class you want to segment, and ./generated.png with the desired output path.

## Pose Editing: 
### PosePerfectT2.ipynb (.ipynb for quicker view of the code and the output), PosePerfectTask2 (to run the code)
What it does: Modifies the rotation of the segmented object.
How it works: Takes the segmented image and user-defined pose parameters (azimuth and polar angles). Azimuth controls rotation around the vertical axis, and polar controls rotation along the object's depth.
Output: A new image with the object's pose adjusted according to the specified angles.

![chair](https://github.com/user-attachments/assets/ee374a3d-8f7c-430e-a3a2-94301abb5a97)
![isolated_chair](https://github.com/user-attachments/assets/a2310af0-6922-494f-8164-12f29c160730)
![rotated_chair (3)](https://github.com/user-attachments/assets/1f350b35-5f22-42e7-a663-05728f09177d)

![highlighted_chair_white](https://github.com/user-attachments/assets/8c7b1ddf-d1a8-401a-bf62-f8f9697fd091)
![replaced_chair](https://github.com/user-attachments/assets/e3047eca-f88a-466d-80c5-a45fbf67f72f)

## SetUp
1. Open VSCode 
2. Copy the given command
```shell
git clone https://github.com/saarinii/PosePerfect.git
```
3. Navigate to the project directory:
```shell
cd PosePerfectTask2
```
4. Install the required packages:
```shell
pip install -r requirements.txt
```
5. You can now run the script as follows::
```shell
python run.py --image ./example.jpg --class "chair" --azimuth +72 --polar +0 --output ./generated.png
```
Replace ./example.jpg with the path to your input image, "chair" with the object class you want to segment, ./generated.png with the desired output path, and '--azimuth +72 --polar +0' with the angle you want to rotate the object to. 
## My trials
To look more into my learning journey and what all the experimenting and trial and errors I had to do to make this project and a lot of my mistakes head to 
### AvatarAssignmentAllMyTrys.ipynb 
