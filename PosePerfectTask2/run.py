import os
import cv2
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Configuration
CHECKPOINT_PATH = '/content/weights/sam_vit_h_4b8939.pth'
MODEL_TYPE = 'vit_h'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load SAM model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def display_image(image_path):
    """Display an image given its path."""
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    else:
        print(f"Error: Image at {image_path} could not be loaded.")

def detect_and_segment(image_path, target_object):
    """Detect the target object and segment it using SAM."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return None

    # Detect objects using YOLOv5
    results = model(img)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    
    # Iterate through detected objects to find the target object
    for label, cord in zip(labels, cords):
        if model.names[int(label)] == target_object:
            x1, y1, x2, y2, conf = cord
            h, w = img.shape[:2]
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

            # Segment the ROI using SAM
            roi = img[y1:y2, x1:x2]
            image_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            sam_result = mask_generator.generate(image_rgb)
            if not sam_result:
                print("No segments found in the ROI.")
                return None

            # Find and apply the largest mask
            largest_mask = max(sam_result, key=lambda mask: np.sum(mask['segmentation']))['segmentation'].astype(np.uint8) * 255
            
            # Process the image
            return process_image(img, largest_mask, x1, y1, x2, y2)

    print(f"No {target_object} detected.")
    return None

def process_image(original_image, mask, x1, y1, x2, y2):
    """Process the image to replace the detected object."""
    full_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask

    # Create a white background for the object
    white_object = np.ones_like(original_image) * 255
    object_area = cv2.bitwise_and(white_object, white_object, mask=full_mask)

    # Combine the white object with the rest of the original image
    rest_of_image = cv2.bitwise_and(original_image, original_image, mask=cv2.bitwise_not(full_mask))
    final_result = cv2.add(object_area, rest_of_image)

    # Save and display the final image
    output_path = './processed_image.jpg'
    cv2.imwrite(output_path, final_result)
    print(f"Processed image saved as '{output_path}'.")

    if os.path.exists(output_path):
        display_image(output_path)

    return output_path

def main():
    parser = argparse.ArgumentParser(description="Detect and replace objects in an image.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--class', type=str, required=True, help="Target object class to detect.")
    parser.add_argument('--output', type=str, default='./generated.png', help="Output path for the generated image.")

    args = parser.parse_args()

    image_path = args.image
    target_object = args.class
    output_path = args.output

    mask_path = detect_and_segment(image_path, target_object)

    if mask_path:
        print(f"Final image saved at: {mask_path}")
    else:
        print("No image was generated.")

if __name__ == "__main__":
    main()
