import torch
import os
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_DIR = "./inputs"
OUTPUT_DIR = "./outputs"
LOCAL_MODEL_PATH = "./BiRefNet_Local"
TARGET_RATIO = 0.8          # 4:5 ratio
INTERNAL_PADDING = 0.10     # 10% padding around the object inside the frame
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Model
model = AutoModelForImageSegmentation.from_pretrained(
    LOCAL_MODEL_PATH, 
    trust_remote_code=True,
    local_files_only=True
).to(device).eval()

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def process_to_portrait_with_crop(img, target_ratio, padding_pct):
    # Step 1: Auto-Crop to object boundaries using the Alpha channel
    bbox = img.getbbox() # Finds the tightest box around non-transparent pixels
    if not bbox:
        return img # Return original if image is empty
    
    # Crop to the object
    obj_only = img.crop(bbox)
    obj_w, obj_h = obj_only.size
    
    # Step 2: Add internal padding so object isn't touching the edges
    pad_w = int(obj_w * padding_pct)
    pad_h = int(obj_h * padding_pct)
    
    # Create a padded version of the object
    padded_obj_w = obj_w + (pad_w * 2)
    padded_obj_h = obj_h + (pad_h * 2)
    padded_obj = Image.new("RGBA", (padded_obj_w, padded_obj_h), (0, 0, 0, 0))
    padded_obj.paste(obj_only, (pad_w, pad_h))
    
    # Step 3: Calculate the Portrait Canvas size
    current_ratio = padded_obj_w / padded_obj_h
    
    if current_ratio > target_ratio:
        # Padded object is too wide -> Make canvas taller
        canvas_h = int(padded_obj_w / target_ratio)
        canvas_w = padded_obj_w
    else:
        # Padded object is already portrait/thin -> Make canvas wider/adjust
        canvas_w = int(padded_obj_h * target_ratio)
        canvas_h = padded_obj_h

    # Step 4: Final Assembly
    final_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    # Center the padded object on the final portrait canvas
    offset = ((canvas_w - padded_obj_w) // 2, (canvas_h - padded_obj_h) // 2)
    final_canvas.paste(padded_obj, offset)
    
    return final_canvas

# 2. Bulk Loop
files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

print(f"Processing {len(files)} images with Auto-Crop...")
with torch.no_grad():
    for filename in tqdm(files):
        try:
            path = os.path.join(INPUT_DIR, filename)
            input_image = Image.open(path).convert("RGB")
            
            # AI Inference
            input_tensor = transform_image(input_image).unsqueeze(0).to(device)
            with torch.cuda.amp.autocast():
                preds = model(input_tensor)[-1].sigmoid().cpu()
            
            # Create/Apply Mask
            mask = transforms.ToPILImage()(preds[0].squeeze())
            mask = mask.resize(input_image.size, Image.BILINEAR)
            input_image.putalpha(mask)
            
            # Apply Auto-Crop and Portrait Logic
            result = process_to_portrait_with_crop(input_image, TARGET_RATIO, INTERNAL_PADDING)
            
            # Save
            result.save(os.path.join(OUTPUT_DIR, filename.split('.')[0] + ".png"))
            
            # Memory Cleanup
            del input_tensor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Failed on {filename}: {e}")

print(f"All done! Files saved in: {OUTPUT_DIR}")