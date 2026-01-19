import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# 1. Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load the model
# We use trust_remote_code=True because BiRefNet uses custom architectural code
model = AutoModelForImageSegmentation.from_pretrained(
    'zhengpeng7/BiRefNet', 
    trust_remote_code=True
)
model.to(device)
model.eval()

# 3. Define Image Preprocessing
# BiRefNet works best with 1024x1024 input
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def remove_background(image_path, output_path="output.png"):
    # Load image
    input_image = Image.open(image_path).convert("RGB")
    original_size = input_image.size
    
    # Preprocess
    input_images = transform_image(input_image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    
    # Post-process mask
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(original_size, Image.BILINEAR)
    
    # Apply mask to create transparent image
    input_image.putalpha(mask)
    input_image.save(output_path)
    print(f"Result saved to {output_path}")

# 4. Run Test
# You can replace this URL with a local path
import requests
from io import BytesIO

url = "D:\\Files\\Projects\\Personal Projects\\Tools\\BG-Remover\\inp3.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img.save("input_test2.jpg")

remove_background("input_test2.jpg", "resul2t.png")