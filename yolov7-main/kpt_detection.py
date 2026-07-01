#kpt_detection.py
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import output_to_keypoint, plot_skeleton_kpts

def load_kpt_model(weights_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load(weights_path, map_location=device, weights_only=False)
    model = weigths['model']
    _ = model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)

    return device,model

def predict_kpt(image, device, model):

    orig_h, orig_w = image.shape[:2]

    img = letterbox(image, 960, stride=64, auto=True, scaleup=False)[0]

    letter_h, letter_w = img.shape[:2]

    img_tensor = transforms.ToTensor()(img)
    img_tensor = torch.tensor([img_tensor.numpy()])

    if torch.cuda.is_available():
        img_tensor = img_tensor.half().to(device)
    else:
        img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output, _ = model(img_tensor)

    output = non_max_suppression_kpt(
        output,
        0.25,
        0.65,
        nc=model.yaml['nc'],
        nkpt=model.yaml['nkpt'],
        kpt_label=True
    )

    output = output_to_keypoint(output)

    if output is None or len(output) == 0:
        return None, image, (letter_h, letter_w)

    scale_x = orig_w / letter_w
    scale_y = orig_h / letter_h

    for i in range(output.shape[0]):

        # bbox coords
        output[i, 2] *= scale_x
        output[i, 3] *= scale_y
        output[i, 4] *= scale_x
        output[i, 5] *= scale_y

        # keypoints
        keypoints = output[i, 7:]

        for j in range(0, len(keypoints), 3):

            keypoints[j] *= scale_x
            keypoints[j + 1] *= scale_y

    return output, image, (letter_h, letter_w)