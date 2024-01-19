import torch
import torchvision.models
import torchvision.transforms as transforms

from reshape import reshape_model

def load_model():
    model = torchvision.models.__dict__[SAVED_PTH['arch']]()
    model = reshape_model(model, SAVED_PTH['num_classes'])
    model.load_state_dict(SAVED_PTH['state_dict'])
    model.eval()
    return model

MODEL_PATH = "original_model_best.pth.tar"
SAVED_PTH = torch.load(MODEL_PATH)
MODEL = load_model()

def create_image_tensor(pil_img, resolution):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
    val_transforms = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        normalize,
    ])

    t = val_transforms(pil_img)

    t = t.unsqueeze(0)

    return t

def inference_image(pil_img):
    img_tensor = create_image_tensor(pil_img, SAVED_PTH['resolution'])

    output = MODEL(img_tensor)
    prediction_idx = torch.argmax(output).numpy()
    confidence = torch.max(torch.nn.Softmax(dim=1)(output)).detach().numpy() * 100

    prediction = SAVED_PTH['classes'][prediction_idx]

    return (prediction, confidence)