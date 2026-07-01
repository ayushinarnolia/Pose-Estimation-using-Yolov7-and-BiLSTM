#kpt.py
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import output_to_keypoint, plot_skeleton_kpts,plot_images, plot_one_box

def load_model(weights_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load(weights_path, map_location=device, weights_only=False)
    model = weigths['model']
    _ = model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)
    return device,model


def process_frame(frame, model, device):

    # 1. Prepare image for inference
    img_padded = letterbox(frame, 960, stride=64, auto=True)[0]
    img_tensor = transforms.ToTensor()(img_padded)
    img_tensor = torch.tensor(np.array([img_tensor.numpy()]))

    if torch.cuda.is_available():
        img_tensor = img_tensor.half().to(device)

    # 2. Model Inference
    with torch.no_grad():
        output, _ = model(img_tensor)

    output = non_max_suppression_kpt(
        output, 0.25, 0.65,
        nc=model.yaml['nc'],
        nkpt=model.yaml['nkpt'],
        kpt_label=True
    )

    boxes = []
    keypoints_list = []

    for det in output:
        if len(det):
            # Scale boxes from 960x960 padded space back to original frame size
            scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape)

            # --- MANUAL KEYPOINT SCALING BYPASS ---
            # --- MATHEMATICAL KEYPOINT SCALING VIA GAIN/PAD BYPASS ---
            kpts_all = det[:, 6:]  # shape: (Num_People, 51)
            num_people = kpts_all.shape[0]

            # Reshape into (Num_People, 17, 3) -> [x, y, confidence]
            kpts_reshaped = kpts_all.view(num_people, 17, 3)

            # Re-calculate the exact scaling gain and padding used by letterbox
            # img_tensor.shape[2:] is the padded size (e.g., 960, 960)
            # frame.shape is the original video frame size (height, width, channels)
            gain = min(img_tensor.shape[2] / frame.shape[0], img_tensor.shape[3] / frame.shape[1])

            # Calculate the exact pixel padding added to the top/bottom or sides
            pad_x = (img_tensor.shape[3] - frame.shape[1] * gain) / 2
            pad_y = (img_tensor.shape[2] - frame.shape[0] * gain) / 2

            # Create a shallow list to accumulate our fixed tensors
            kpts_scaled_list = []

            for p in range(num_people):
                person_kpts = kpts_reshaped[p]
                xy_coords = person_kpts[:, :2].clone()

                # Apply reverse scaling transformation formulas:
                # 1. Subtract the letterbox padding margins from the positions
                # 2. Divide by the proportional aspect ratio scale gain
                xy_coords[:, 0] = (xy_coords[:, 0] - pad_x) / gain  # Fix X positions
                xy_coords[:, 1] = (xy_coords[:, 1] - pad_y) / gain  # Fix Y positions

                # Re-attach the untouched visibility confidence score column
                conf_scores = person_kpts[:, 2:3]
                scaled_person = torch.cat([xy_coords, conf_scores], dim=1)  # shape: (17, 3)

                kpts_scaled_list.append(scaled_person)

            # Overwrite the prediction tensor matrix slice with perfectly scaled numbers
            if num_people > 0:
                kpts_recombined = torch.stack(kpts_scaled_list, dim=0)
                det[:, 6:] = kpts_recombined.view(num_people, -1)
            # ---------------------------------------------------------

            det = det.detach().cpu().numpy()

            for d in det:
                # Box format: Convert x1, y1, x2, y2 coordinates into standard rounded integers
                boxes.append([
                    int(round(d[0])),
                    int(round(d[1])),
                    int(round(d[2])),
                    int(round(d[3])),
                    float(d[4])  # Keep confidence score as a decimal float
                ])

                # Extract the 17 keypoint slices for the embedding utility
                kpts_raw = d[6:]
                individual_kpts = kpts_raw.reshape(-1, 3)
                xy_only = individual_kpts[:, :2]  # Keeps raw X, Y

                # Convert coordinate positions to rounded integers for the plot function
                xy_only_int = np.round(xy_only).astype(np.int32)
                keypoints_list.append(xy_only_int)

    return boxes, keypoints_list


# device,model=load_model(r'D:\Projects\Keypoints\yolov7-main\yolov7-w6-pose.pt')
# img=cv2.imread(r'D:\Projects\Keypoints\yolov7-main\standing group.jpg')
# boxes,keypoints_list=process_frame(img,model,device)
# print(boxes)
# print(keypoints_list)

# Create a clean copy of your original image to draw on
# visualization_img = img.copy()
#
# print(f"Verifying outputs... Total people detected: {len(boxes)}")
#
# # 1. Draw the bounding boxes using your returned variables
# for i in range(len(boxes)):
#     current_box = boxes[i]
#     xyxy = current_box[:4]
#     confidence = current_box[4]
#
#     label = f"Person {confidence:.2f}"
#     plot_one_box(
#         xyxy,
#         visualization_img,
#         label=label,
#         color=(0, 255, 0),
#         line_thickness=2
#     )
#
# # 2. Draw the full skeleton keypoints correctly
# for i in range(len(keypoints_list)):
#     current_kpts = keypoints_list[i]  # Array of shape (17, 2)
#
#     # Create a matching column of dummy confidence scores (1.0)
#     # Shape transitions from (17, 2) to (17, 3) -> [x, y, visibility]
#     dummy_confidence = np.ones((17, 1), dtype=np.float32)
#     kpts_with_conf = np.hstack([current_kpts, dummy_confidence])
#
#     # Flatten the array to a continuous 1D row vector (51 elements)
#     # This matches exactly what plot_skeleton_kpts iterates over internally
#     kpts_flat = kpts_with_conf.flatten()
#
#     # Pass all 17 joints together in one call
#     plot_skeleton_kpts(visualization_img, kpts_flat, steps=3)
#
# # 3. Display the final verified canvas
# cv2.imshow("Output Verification", visualization_img)
# print("Press any key inside the image window to close it...")
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# image = cv2.imread(r'D:\Projects\Keypoints\yolov7-main\standing.jpg')
# img=image.copy()
# image = letterbox(image, 960, stride=64, auto=True)[0]
# image_ = image.copy()
# image = transforms.ToTensor()(image)
# image = torch.tensor(np.array([image.numpy()]))
#
# if torch.cuda.is_available():
#     image = image.half().to(device)
# output, _ = model(image)
#
# output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
#
# nimg = image[0].permute(1, 2, 0) * 255
# nimg = nimg.cpu().numpy().astype(np.uint8)
# nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
#
# for det in output:  # one image at a time
#     if len(det):
#         for d in det:  # one object
#             d = d.detach().cpu().numpy()
#             xyxy= d[:4]
#             conf = d[4]
#             label = f'person {conf:.2f}'
#             print(img)
#             plot_one_box(
#                 xyxy,
#                 nimg,
#                 label=label,
#                 color=(0, 255, 0),
#                 line_thickness=2
#             )
#
# with torch.no_grad():
#     output = output_to_keypoint(output)
#
# for idx in range(output.shape[0]):
#     plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
#
# nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
#
# cv2.imshow("Pose Estimation", nimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()