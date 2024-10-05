import argparse
import torch
import os
import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from diffusers import StableDiffusionInpaintPipeline

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Device configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Function to detect object using YOLOv5
def detect_and_segment(image_path, target_object, sam_checkpoint, model_type):
    # Load the image
    img = cv2.imread(image_path)
    
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
            
            # Segment the ROI using SAM
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(DEVICE)
            mask_generator = SamAutomaticMaskGenerator(sam)
            masks = mask_generator.generate(roi)
            
            # Select the mask with the highest score (confidence)
            best_mask = max(masks, key=lambda x: x['score'])
            segmented_img = (best_mask['segmentation'] * 255).astype(np.uint8)
            return segmented_img, (x1, y1, x2, y2)

    if not object_detected:
        raise ValueError(f"{target_object} not detected in the image.")
    
# Function for inpainting using Diffusers
def inpaint_with_diffusers(image_path, mask, output_path):
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16).to(DEVICE)
    
    # Load the input image and mask
    image = Image.open(image_path).convert("RGB")
    mask_image = Image.fromarray(mask).convert("L")
    
    # Inpaint the image
    result = pipe(prompt="inpainting", image=image, mask_image=mask_image).images[0]
    
    # Save the output
    result.save(output_path)
    print(f"Inpainted image saved at: {output_path}")

# Argument parser to handle inputs
def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection, Segmentation, and Inpainting Script")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--class', type=str, required=True, help='Class of the object to detect (e.g., chair).')
    parser.add_argument('--azimuth', type=int, required=True, help='Azimuth angle adjustment (e.g., +72).')
    parser.add_argument('--polar', type=int, required=True, help='Polar angle adjustment (e.g., +0).')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output image.')
    parser.add_argument('--sam_checkpoint', type=str, required=True, help='Path to the SAM model checkpoint.')
    parser.add_argument('--model_type', type=str, default="vit_h", help='SAM model type (e.g., vit_h).')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Detect and segment the target object
    segmented_img, coords = detect_and_segment(args.image, args.class, args.sam_checkpoint, args.model_type)
    
    # Save the segmented image for visualization (optional)
    segmented_output = args.output.replace('.png', '_segmented.png')
    cv2.imwrite(segmented_output, segmented_img)
    
    # Perform inpainting on the segmented image
    inpaint_with_diffusers(args.image, segmented_img, args.output)

if __name__ == "__main__":
    main()
