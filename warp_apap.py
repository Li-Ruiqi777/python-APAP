import cv2
import numpy as np
import matchers
from ransac import RANSAC
from apap import APAP_stitching, get_mdlt_final_size
from imagewarping import imagewarping
import config

from utils.logger_config import *
logger = logging.getLogger(__name__)

def compute_homography_APAP(ref_img, tar_img, visualize=False):
    """
    计算从目标图像到参考图像的单应性矩阵H (使用SIFT特征)
    参数：
        ref_img: 参考图像(BGR格式)
        tar_img: 目标图像(BGR格式)
    返回：
        warped_target_img: 变换后的目标图像
        warped mask: 变换后的mask(用于求重叠部分)
    """
    ## SIFT keypoint detection and matching
    matcher_obj = matchers.matchers()
    kp1, ds1 = matcher_obj.getFeatures(ref_img)
    kp2, ds2 = matcher_obj.getFeatures(tar_img)
    matches = matcher_obj.match(ds1, ds2)

    src_orig = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_orig = np.float32([kp2[m.trainIdx].pt for m in matches])
    src_orig = np.vstack((src_orig.T, np.ones((1, len(matches)))))
    dst_orig = np.vstack((dst_orig.T, np.ones((1, len(matches)))))

    ##################
    # Outlier removal.
    ##################
    ransac = RANSAC(config.M, config.thr, visual=False)
    try:
        src_fine, dst_fine = ransac(ref_img, tar_img, src_orig, dst_orig)
    except Exception as e:
        logger.info(f"RANSAC failed: {e}")
        return ref_img, tar_img, np.ones_like(ref_img)
    src_fine, dst_fine = ransac(ref_img, tar_img, src_orig, dst_orig)

    ##########################
    # Moving DLT (projective).
    ##########################
    # Generating mesh for MDLT
    X, Y = np.meshgrid(np.linspace(0, tar_img.shape[1]-1, config.C1), np.linspace(0, tar_img.shape[0]-1, config.C2))
    # Mesh (cells) vertices' coordinates
    Mv = np.array([X.ravel(), Y.ravel()]).T
    # Perform Moving DLT
    apap = APAP_stitching(config.gamma, config.sigma)
    Hmdlt = apap(dst_fine, src_fine, Mv)

    ##################################
    # Image stitching with Moving DLT.
    ##################################
    min_x, max_x, min_y, max_y = get_mdlt_final_size(ref_img, tar_img, Hmdlt, config.C1, config.C2)
    limit = 2500
    if min_x<(-limit) or max_x > limit or min_y<(-limit) or max_y > limit:
        return ref_img, tar_img, np.ones_like(ref_img)
    warped_ref_img, warped_tar_img, warped_ref_mask, warped_tar_mask = imagewarping(ref_img, tar_img, Hmdlt, min_x, max_x, min_y, max_y,
                                                                        config.C1, config.C2)
    if warped_ref_img is None or warped_tar_img is None or warped_ref_mask is None or warped_tar_mask is None:
        return ref_img, tar_img, np.ones_like(ref_img)
    
    # linear_mdlt = imageblending(warped_ref_img, warped_tar_img, warped_ref_mask, warped_tar_mask)
    warped_tar_mask = np.stack([warped_tar_mask] * 3, axis=-1)/255
    warped_ref_mask = np.stack([warped_ref_mask] * 3, axis=-1)/255

    overlap_mask = np.logical_and(warped_ref_mask, warped_tar_mask)

    if visualize:
        cv2.imshow("warped_target_img", warped_tar_img * overlap_mask)
        cv2.imshow("warped_tar_mask", warped_tar_mask)
        cv2.imshow("warped_ref_img", warped_ref_img * overlap_mask)
        cv2.imshow("warped_ref_mask", warped_ref_mask)
        # cv2.imshow("linear_mdlt", linear_mdlt)
        cv2.waitKey(0)

    return warped_ref_img, warped_tar_img, overlap_mask


if __name__ == "__main__":
    ref_img = cv2.imread('E:/DeepLearning/0_DataSets/007-UDIS-D/testing/input1/000001.jpg')
    target_img = cv2.imread('E:/DeepLearning/0_DataSets/007-UDIS-D/testing/input2/000001.jpg')

    # 执行配准
    warped_target_img, H = compute_homography_APAP(ref_img, target_img, True)
