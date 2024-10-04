# PosePerfect
PosePerfect is a Python project that allows you to edit the pose of an object within a scene using cutting-edge generative AI techniques. This project focuses on two key tasks:

## Object Segmentation: PosePerfectT1.ipynb

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



## Pose Editing:

What it does: Modifies the rotation of the segmented object.
How it works: Takes the segmented image and user-defined pose parameters (azimuth and polar angles). Azimuth controls rotation around the vertical axis, and polar controls rotation along the object's depth.
Output: A new image with the object's pose adjusted according to the specified angles.
