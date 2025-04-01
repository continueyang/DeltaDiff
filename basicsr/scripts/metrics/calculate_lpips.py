import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def main():
    # Configurations
    # -------------------------------------------------------------------------
    folder_gt = "data/CELEBA/TEST_HR/"
    folder_restored = "results/gvtv9x8/celeba275000"
    crop_border = 8
    suffix = ''
    # -------------------------------------------------------------------------
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))

    mean = [0.5430, 0.4218, 0.3605]
    std = [0.07787433 ,0.07787433, 0.07787433]
    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(osp.join(folder_restored, basename +'_SwinIR'+ ext), cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())

        print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val.item():.6f}.')
        lpips_all.append(lpips_val)
        average_lpips = sum(lpips_all).item() / len(lpips_all)
    print(f'Average: LPIPS: {average_lpips:.6f}')


if __name__ == '__main__':
    main()
