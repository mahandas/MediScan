from torch import nn
import torch
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from PIL import ImageFont
from uuid import uuid4
import os


UPLOAD_FOLDER = "uploads"
DEVICE = "cpu"

MAMOGRAM_MODEL = "./models/breast1.pt"

font = ImageFont.truetype('/Users/mahdas/Downloads/ARIAL.TTF', 30) 


def transform_image(image, my_transforms=None):
    if my_transforms:
        return my_transforms(image).unsqueeze(0)
    return transforms.ToTensor()(image).unsqueeze(0)


def get_model(num_classes):   
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features   
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

    return model

obj_detection_model = get_model(num_classes=2)
obj_detection_model.load_state_dict(torch.load(MAMOGRAM_MODEL), map_location='cpu')
obj_detection_model = obj_detection_model.to(DEVICE)
obj_detection_model.eval()



def predict_mamogram(path):
    filename = str(uuid4()) + '.jpg' 
    image = Image.open(path).convert('rgb')
    obj_tensor = transform_image(image)
    obj_tensor = obj_tensor.to(DEVICE)
    pil_img = transforms.ToPILImage()(obj_tensor.squeeze(0))
    draw = ImageDraw.Draw(pil_img)
    prediction = obj_detection_model(obj_tensor)[0]
    for element in range(len(prediction["boxes"])):
        boxes = prediction["boxes"][element].cpu().numpy()
        score = np.round(prediction["scores"][element].cpu().numpy(),
            decimals= 4)
        if score > threshold:
            draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], 
                outline ="red", width =3)
            draw.text((boxes[0], boxes[1]), text = str(score), font=font)
            pil_img.save(os.path.join(UPLOAD_FOLDER, filename))
    return filename



