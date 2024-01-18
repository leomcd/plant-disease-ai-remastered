import argparse

import torch
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image

import time

from reshape import reshape_model

parser = argparse.ArgumentParser(description='Run Image Classifier')

parser.add_argument('--img', type=str, default='data/test/CornCommonRust1.JPG', 
                    help='path to desired image to run model on'
					'image path (default: data/test/PotatoHealthy1.JPG/)')

args = parser.parse_args()

#TODO: put the god forsaken global model variable so you dont need to reload the model everytime

def create_image_tensor(img_path, resolution):
    img = Image.open(img_path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
    val_transforms = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        normalize,
    ])

    t = val_transforms(img)

    t = t.unsqueeze(0)

    return t

def inference_image(img):
    saved_pth = torch.load("models34/model_best.pth.tar")

    model = torchvision.models.__dict__[saved_pth['arch']]()

    model = reshape_model(model, saved_pth['num_classes'])
    model.load_state_dict(saved_pth['state_dict'])
    model.eval()

    then = time.time()

    img_tensor = create_image_tensor(img, saved_pth['resolution'])

    output = model(img_tensor)
    prediction_idx = torch.argmax(output).numpy()
    confidence = torch.max(torch.nn.Softmax(dim=1)(output)).detach().numpy() * 100

    prediction = saved_pth['classes'][prediction_idx]

    now = time.time()

    print(now-then)

    return (prediction, confidence)

if __name__ == "__main__":
    print(inference_image(args.img))