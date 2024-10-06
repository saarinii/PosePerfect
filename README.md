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
![highlighted_chair_red](https://github.com/user-attachments/assets/5d895fc3-542d-4361-be6f-b4c483629832)


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
What it does:
Detects a target object in an image, segments it, and then uses inpainting to modify the image based on the mask of the detected object.

How it works:
The script accepts an input image and a text prompt specifying the object class (e.g., chair, table). It uses YOLOv5 to detect the specified object, SAM to generate a mask around the object, and Stable Diffusion inpainting to modify the image based on the mask.

Output:
An inpainted image where the specified object is detected, segmented, and modified (e.g., filled in or removed) using Stable Diffusion. The output is saved as a new image file.

### Key Components

#### Library Imports:

OpenCV (cv2): For image processing (reading, writing, and displaying).
NumPy (np): For numerical operations on image data.
Torch: For loading and running the YOLOv5, SAM, and Stable Diffusion models.
PIL (Pillow): For handling images in the inpainting process.
Diffusers: To handle the Stable Diffusion inpainting pipeline.

#### Configuration:

The script checks if CUDA is available to run models on a GPU.
It specifies a pre-trained YOLOv5 model from Ultralytics for object detection.
The Segment Anything Model (SAM) is loaded using a checkpoint, with its type (e.g., vit_h) specified.

#### Models:

YOLOv5: Loaded from Ultralytics, this pre-trained model detects objects in the input image and extracts the region of interest (ROI) for further segmentation.
SAM (Segment Anything Model): SAM is used to generate a mask around the detected object, allowing for precise segmentation.
Stable Diffusion Inpainting: Uses the diffusers library to apply inpainting on the segmented area, modifying the image based on the generated mask.
Detection and Segmentation Function:

Detecting Objects:
The script reads an image and detects objects using YOLOv5, identifying the specified object class by its label (e.g., chair). The detected object's coordinates are used to extract the ROI.

Segmentation:
SAM is employed to generate a mask for the object within the ROI, and the best mask (highest confidence) is selected for further processing.

Inpainting:
The mask is applied to the image, and Stable Diffusion is used to perform inpainting on the segmented region. This modifies the image based on the prompt and mask (e.g., removing or replacing the object).

#### Image Saving and Output:
The script saves the segmented mask and the final inpainted image as separate output files, ensuring you get both the segmented and modified versions of the input image.


![chair](https://github.com/user-attachments/assets/ee374a3d-8f7c-430e-a3a2-94301abb5a97)
![isolated_chair](https://github.com/user-attachments/assets/a2310af0-6922-494f-8164-12f29c160730)
![rotated_chair (3)](https://github.com/user-attachments/assets/1f350b35-5f22-42e7-a663-05728f09177d)

![highlighted_chair_white](https://github.com/user-attachments/assets/8c7b1ddf-d1a8-401a-bf62-f8f9697fd091)
![replaced_chair (1)](https://github.com/user-attachments/assets/1b31d309-7d56-4950-bb4a-3f894979c105)


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
