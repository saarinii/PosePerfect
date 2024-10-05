import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to display the image
def display_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        plt.imshow(image)
        plt.axis('off')  # Hide axis
        plt.show()
    else:
        print(f"Error: Image at {image_path} could not be loaded.")

# Function to detect an object, segment it using SAM, and save the image with the red mask
def detect_and_segment(image_path, target_object, output_path):
    # Configuration
    CHECKPOINT_PATH = './weights/sam_vit_h_4b8939.pth'  # Path to your SAM model weights
    MODEL_TYPE = 'vit_h'  # Specify your SAM model type
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

    # Load SAM model
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return

    # Detect objects using YOLOv5
    results = model(img)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    object_detected = False

    # Iterate through detected objects to find the target object
    for label, cord in zip(labels, cords):
        if model.names[int(label)] == target_object:
            object_detected = True
            x1, y1, x2, y2, conf = cord
            print(f"{target_object.capitalize()} detected with confidence {conf:.2f} at location: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")

            # Extract ROI
            h, w = img.shape[:2]
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            roi = img[y1:y2, x1:x2]
            print("Region of Interest (ROI) extracted.")

            # Segment the ROI using SAM
            image_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            sam_result = mask_generator.generate(image_rgb)
            if not sam_result:
                print("No segments found in the ROI.")
                return

            # Display the number of segments found
            print(f"Found {len(sam_result)} segments.")

            # Find and apply the largest mask
            largest_mask = None
            max_area = 0
            for mask in sam_result:
                segmentation = mask['segmentation']
                area = np.sum(segmentation)
                if area > max_area:
                    max_area = area
                    largest_mask = segmentation.astype(np.uint8) * 255

            print(f"Applying the largest mask with area {max_area}.")

            # Create a red highlight for the segmented object
            red_highlight = np.zeros_like(roi)
            red_highlight[:, :, 2] = 255  # Set Red channel to 255

            # Blend the original ROI with the red highlight using the mask
            highlighted_roi = cv2.bitwise_and(red_highlight, red_highlight, mask=largest_mask)
            highlighted_roi = cv2.addWeighted(highlighted_roi, 0.5, roi, 0.5, 0)

            # Replace the ROI in the original image with the highlighted version
            img[y1:y2, x1:x2] = highlighted_roi

            # Save the final image with red-highlighted object
            cv2.imwrite(output_path, img)
            print(f"Image with segmented object saved as '{output_path}'.")

            break

    if not object_detected:
        print(f"No {target_object} detected.")

# Main function to handle command-line arguments and run the process
def main():
    parser = argparse.ArgumentParser(description="Object Detection and Segmentation with Red Mask")
    parser.add_argument('--image', required=True, help="Path to the input image")
    parser.add_argument('--class', required=True, help="Object class to detect (e.g., 'chair')")
    parser.add_argument('--output', required=True, help="Path to save the output image with red mask")
    
    args = parser.parse_args()

    # Call the detection and segmentation function
    detect_and_segment(args.image, args.class, args.output)

if __name__ == "__main__":
    main()
