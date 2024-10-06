import os
import cv2
import numpy as np
import torch
from PIL import Image
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline

# Define device and models globally
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = './weights/sam_vit_h_4b8939.pth'
MODEL_TYPE = "vit_h"

# Load models only once
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
sam_model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam_model)

# Function to detect objects and segment using SAM
def detect_and_segment(image_path, target_object):
    img = cv2.imread(image_path)
    results = yolo_model(img)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    h, w = img.shape[:2]
    for label, cord in zip(labels, cords):
        if yolo_model.names[int(label)] == target_object:
            x1, y1, x2, y2, conf = cord
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            roi = img[y1:y2, x1:x2]

            # Segment ROI using SAM
            image_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            sam_result = mask_generator.generate(image_rgb)
            if not sam_result:
                return None

            largest_mask = max(sam_result, key=lambda mask: np.sum(mask['segmentation']))
            mask = largest_mask['segmentation'].astype(np.uint8) * 255

            # Apply mask
            segmented_object = cv2.bitwise_and(roi, roi, mask=mask)
            return segmented_object

    return None

# Function to generate rotated image using diffusion model
def rotate_object(image_path, azimuth, polar, output_path):
    pipe = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.1", 
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing='trailing'
    )
    pipe.to(DEVICE)

    cond_image = Image.open(image_path).convert("RGBA")
    prompt = f"Change pose of the object by azimuth {azimuth} degrees; polar {polar} degrees."

    result = pipe(cond_image, prompt=prompt, num_inference_steps=28).images[0]
    result.save(output_path)

# Function to perform inpainting using Stable Diffusion
def inpaint_image(input_image_path, mask_image_path, output_path):
    input_image = Image.open(input_image_path).convert("RGB")
    mask_image = Image.open(mask_image_path).convert("L")  # Convert mask to grayscale
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(DEVICE)

    result_image = pipe(
        prompt="Blend the area around the object seamlessly into the background to create a natural and cohesive appearance.",
        image=input_image,
        mask_image=mask_image,
        num_inference_steps=1500
    ).images[0]

    result_image.save(output_path)

# Main function to handle command-line arguments
def main(args):
    input_image_path = args.image
    target_class = args.class_name
    azimuth = args.azimuth
    polar = args.polar
    output_image_path = args.output
    inpaint = args.inpaint

    print(f"Processing image: {input_image_path}, looking for: {target_class}")

    # Detect and segment the object
    segmented_object = detect_and_segment(input_image_path, target_class)
    if segmented_object is not None:
        segmented_path = "./segmented_object.png"
        cv2.imwrite(segmented_path, segmented_object)

        # Rotate the object using the specified azimuth and polar angles
        rotate_object(segmented_path, azimuth, polar, output_image_path)
        print(f"Rotated object saved to {output_image_path}")

        # Optionally perform inpainting
        if inpaint:
            inpaint_image(input_image_path, segmented_path, output_image_path)
            print(f"Inpainted image saved to {output_image_path}")
    else:
        print(f"No {target_class} detected in {input_image_path}.")

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection, Segmentation, Rotation, and Inpainting")
    parser.add_argument('--image', required=True, help="Path to input image")
    parser.add_argument('--class', dest="class_name", required=True, help="Object class to detect (e.g., 'chair')")
    parser.add_argument('--azimuth', type=int, required=True, help="Azimuth angle for rotation")
    parser.add_argument('--polar', type=int, required=True, help="Polar angle for rotation")
    parser.add_argument('--output', required=True, help="Path to save the generated image")
    parser.add_argument('--inpaint', action='store_true', help="Include this flag to apply inpainting after rotation")
    args = parser.parse_args()
    
    main(args)
