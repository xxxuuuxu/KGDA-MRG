import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from gradcam_utils import GradCAM, LayerCAM, show_cam_on_image
from gradcam_utils import UQGradCAMPP, StandardGradCAM
from model import resnet34
import pickle
import cv2
import glob
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
SAVE_VIS = False


def get_model():
    model = resnet34(num_classes=14).cuda()
    model_weight_path = "./weights/MIMIC_best_weight.pth"
    state_dict = torch.load(model_weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.eval()

def main(model, image_path, seg_path, mask_path, array_path, thresholds, 
         cam_vis_path=None, unc_vis_path=None):


    target_layers = [model.layer4]

    data_transform = transforms.Compose([
        transforms.Resize(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    assert os.path.exists(image_path), "file: '{}' does not exist.".format(image_path)


    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img, dtype=np.uint8)

    orig_w, orig_h = img.size 

    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0).cuda()

    with torch.no_grad():
        logit = model(input_tensor)
    
    thresholded_predictions = 1 * (logit.detach().cpu().numpy() > thresholds)
    indices = np.where(thresholded_predictions[0] == 1)[0]


    cam = UQGradCAMPP(model=model, target_layers=[model.layer4[-1]], use_cuda=True, mc_samples=15)

    mask_arr_ass = np.zeros((300, 300), dtype=np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    aggregated_cam_map = np.zeros((300, 300), dtype=np.float32)
    aggregated_unc_map = np.zeros((300, 300), dtype=np.float32)

    if len(indices) == 0:
        np.save(array_path, mask_arr_ass.astype(bool))

        return
    
    # 循环遍历所有阳性标签
    for target_category in list(indices):

        grayscale_cam_mean, uncertainty_norm = cam(input_tensor=input_tensor, target_category=int(target_category))

        grayscale_cam_mean = grayscale_cam_mean.squeeze() 
        uncertainty_norm = uncertainty_norm.squeeze()
        

        if grayscale_cam_mean.shape != (300, 300):
            grayscale_cam_mean = cv2.resize(grayscale_cam_mean, (300, 300))
        
        if uncertainty_norm.shape != (300, 300):
            uncertainty_norm = cv2.resize(uncertainty_norm, (300, 300))


        aggregated_cam_map = np.maximum(aggregated_cam_map, grayscale_cam_mean)
        aggregated_unc_map = np.maximum(aggregated_unc_map, uncertainty_norm)

        weighted_cam = grayscale_cam_mean * (1.0 - uncertainty_norm)

        heatmap = weighted_cam

        threshold = 0.6
        mask = cv2.threshold(heatmap, threshold, 1, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(mask, kernel)

        mask = cv2.resize(mask.astype(np.float32), (300, 300))
        mask_arr_ass += mask


    mask_arr_ass_bool = mask_arr_ass > 0
    np.save(array_path, mask_arr_ass_bool)

    if True:
        mask_img = Image.fromarray((mask_arr_ass_bool * 255).astype(np.uint8)).convert('L')
        mask_img.save(mask_path)

        img_for_seg = Image.open(image_path).convert('RGB')
        img_for_seg_np = np.asarray(img_for_seg)
        mask_resized = mask_img.resize((img_for_seg_np.shape[1], img_for_seg_np.shape[0]), Image.NEAREST)
        mask_np = np.asarray(mask_resized).astype(np.uint8)
        mask_3ch = np.repeat(mask_np[..., None], 3, axis=2)
        masked_img_np = (img_for_seg_np * (mask_3ch / 255)).astype(np.uint8)
        masked_img = Image.fromarray(masked_img_np)
        masked_img.save(seg_path)


    if cam_vis_path is not None:

        cam_resized = cv2.resize(aggregated_cam_map, (orig_w, orig_h))
        

        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        

        img_rgb_norm = img_np.astype(np.float32) / 255.0

        cam_vis_img = show_cam_on_image(img_rgb_norm, cam_resized, use_rgb=True)


        if isinstance(cam_vis_img, tuple):

            cam_vis_img = cam_vis_img[0]

        if cam_vis_img.max() <= 1.0:
             cam_vis_img = (cam_vis_img * 255).astype(np.uint8)
        
        Image.fromarray(cam_vis_img.astype(np.uint8)).save(cam_vis_path)
        print(f"Saved CAM visualization to {cam_vis_path}")

    if unc_vis_path is not None:

        unc_resized = cv2.resize(aggregated_unc_map, (orig_w, orig_h))

        unc_norm = (unc_resized - unc_resized.min()) / (unc_resized.max() - unc_resized.min() + 1e-8)

        unc_uint8 = (unc_norm * 255).astype(np.uint8)
        

        unc_heatmap = cv2.applyColorMap(unc_uint8, cv2.COLORMAP_JET)

        unc_heatmap = cv2.cvtColor(unc_heatmap, cv2.COLOR_BGR2RGB)

        Image.fromarray(unc_heatmap).save(unc_vis_path)
        print(f"Saved Uncertainty visualization to {unc_vis_path}")


if __name__ == '__main__':
    image_list = glob.glob("/root/autodl-tmp/KGDA-MRG/data/mimic_cxr/images300/*/*/*/*.jpg")
    # image_list = glob.glob("/root/autodl-tmp/KGDA-MRG/data/iu_xray/images/*/*.png")
    model = get_model()

    with open("/root/autodl-tmp/KGDA-MRG/datasets/thresholds.pkl", "rb") as f:
        thresholds = pickle.load(f)

    bar = tqdm.tqdm(image_list)
    for image_path in bar:
        seg_path = image_path.replace("images", "la_images300_seg")
        mask_path = image_path.replace("images", "la_images300_mask")
        array_path = image_path.replace("images", "la_images300_array").replace(".png", ".npy")

        cam_vis_path = image_path.replace("images", "la_images300_cam_vis")
        unc_vis_path = image_path.replace("images", "la_images300_unc_vis")

        os.makedirs(os.path.dirname(seg_path), exist_ok=True)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(array_path), exist_ok=True)
        os.makedirs(os.path.dirname(cam_vis_path), exist_ok=True)
        os.makedirs(os.path.dirname(unc_vis_path), exist_ok=True)



        if not (os.path.exists(array_path) and os.path.exists(cam_vis_path)):
            main(model, 
                 image_path, 
                 seg_path, 
                 mask_path, 
                 array_path, 
                 thresholds,
                 cam_vis_path=cam_vis_path,
                 unc_vis_path=unc_vis_path
            )

