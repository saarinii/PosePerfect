import cv2
import numpy as np
import torch
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os

# Configuration
CHECKPOINT_PATH = '/content/weights/sam_vit_h_4b8939.pth'  # Path to your SAM model
MODEL_TYPE = 'vit_h'  # Specify your SAM model type
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

# Load SAM model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_and_segment(image_path, target_object, output_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return None

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
                return None

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

            # Resize the largest mask to fit the original image
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = largest_mask

            # Invert the mask to isolate the object
            inverted_mask = cv2.bitwise_not(full_mask)

            # Create a red background for the object
            red_mask = np.zeros_like(img)  # Create a black image
            red_mask[:] = [0, 0, 255]  # Fill it with red color

            # Apply the red to the object and leave the rest of the image intact
            object_red = cv2.bitwise_and(red_mask, red_mask, mask=full_mask)
            rest_of_image = cv2.bitwise_and(img, img, mask=inverted_mask)

            # Combine the red object with the rest of the original image
            final_result = cv2.add(object_red, rest_of_image)

            # Save the final image with the red mask
            cv2.imwrite(output_path, final_result)
            print(f"Image with the {target_object} highlighted in red saved as '{output_path}'.")

            return output_path

    if not object_detected:
        print(f"No {target_object} detected.")
        return None

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Highlight an object in an image with a red mask.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--class", required=True, help="Class name of the object to highlight.")
    parser.add_argument("--output", required=True, help="Path to save the output image.")

    # Parse the command line arguments
    args = parser.parse_args()

    # Run the detection and segmentation
    mask_path = detect_and_segment(args.image, args.class, args.output)

    if mask_path:
        print(f"Final image saved at: {mask_path}")
    else:
        print("No image was generated.")
