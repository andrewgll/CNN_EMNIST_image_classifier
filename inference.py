from CNN import Net
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse

out_file = "model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #GPU much faster

print(f"Device will be used: {device}")
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input file path')
args = parser.parse_args()
root_path = args.input

if root_path == None:
    raise ValueError("No input file path provided.")

#Load the saved weights
model = Net()
print("Compiling model, this can take time...")
model = torch.compile(model) #More time to compile but faster execution
model.load_state_dict(torch.load(out_file,map_location=torch.device(device)))
model.eval()


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Lambda(lambda x:Image.fromarray(np.transpose(x))),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#Loop over all the images in the root folder 
files = sorted(os.listdir(root_path))
for file in files:
    if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')):
        # Load the image
        image_path = os.path.join(root_path, file)
        image = Image.open(image_path)

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.argmax(dim=1)
        if prediction.item() > 9:
            print('%03d, %s' % (prediction.item()+55, image_path))
        else:
            print('%03d, %s' % (prediction.item()+48, image_path))

