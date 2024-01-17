import argparse

import torch
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image

from defaultargs import defaultargs

from reshape import reshape_model

parser = argparse.ArgumentParser(description='Run Image Classifier')

parser.add_argument('--img', type=str, default='data/test/AppleScab1.JPG', 
                    help='path to desired image to run model on'
					'image path (default: data/test/AppleScab1.JPG/)')

args = parser.parse_args()

for key in defaultargs.keys():
    args.__dict__[key] = defaultargs[key]

def create_image_tensor(img_path):
    img = Image.open(img_path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
    val_transforms = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        normalize,
    ])

    t = val_transforms(img)

    t = t.unsqueeze(0)

    return t

def inference_image(img):
    img_tensor = create_image_tensor(img)

    model = torchvision.models.__dict__[args.arch]()

    saved_pth = torch.load("models/model_best.pth.tar")

    model = reshape_model(model, saved_pth['num_classes'])
    model.load_state_dict(saved_pth['state_dict'])
    model.eval()

    output = model(img_tensor)
    prediction_idx = torch.argmax(output).numpy()
    confidence = torch.max(torch.nn.Softmax(dim=1)(output)).detach().numpy() * 100

    prediction = saved_pth['classes'][prediction_idx]

    return (prediction, confidence)

if __name__ == "__main__":
    print(inference_image(args.img))