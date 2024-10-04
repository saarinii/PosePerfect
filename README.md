# PosePerfect
PosePerfect is a Python project that allows you to edit the pose of an object within a scene using cutting-edge generative AI techniques. This project focuses on two key tasks:

## Object Segmentation:

What it does: Isolates the target object from the background image.
How it works: Takes an input image and a text prompt specifying the object class (e.g., chair, table).
Output: A segmented image with a red mask highlighting the object's boundaries.


## Pose Editing:

What it does: Modifies the rotation of the segmented object.
How it works: Takes the segmented image and user-defined pose parameters (azimuth and polar angles). Azimuth controls rotation around the vertical axis, and polar controls rotation along the object's depth.
Output: A new image with the object's pose adjusted according to the specified angles.
