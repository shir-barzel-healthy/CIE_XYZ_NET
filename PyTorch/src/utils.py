"""
 Copyright 2020 Mahmoud Afifi.
 Released under the MIT License.
 If you use this code, please cite the following paper:
 Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith
 Punnappurath, and Michael S Brown.
 CIE XYZ Net: Unprocessing Images for Low-Level Computer Vision Tasks.
 arXiv preprint, 2020.
"""

__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from math import sqrt,log10

def PSNR(original, recon):
    if type(original).__module__ != np.__name__:
        original = np.array(original.cpu())
        recon = np.array(recon.cpu())
    mse = np.mean((original - recon) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255 ##np.max(original)  ## TODO
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I


def compute_loss_self_sup(imgs, xyz_gt, rec_1, rend_1, m_inv_1,
                 m_fwd_1, rec_2, rend_2, m_inv_2, m_fwd_2):
    srgb_gt_1 = torch.squeeze(imgs[:, 0, :, :, :])
    srgb_gt_2 = torch.squeeze(imgs[:, 1, :, :, :])

    xyz_gt_1 = torch.squeeze(xyz_gt[:, 0, :, :, :])
    xyz_gt_2 = torch.squeeze(xyz_gt[:, 1, :, :, :])

    loss_m_inv = torch.norm(m_inv_1 - m_inv_2, p='fro', dim=[1, 2]) ## TODO optimization - rewrite norm
    loss_m_fwd = torch.norm(m_fwd_1 - m_fwd_2, p='fro', dim=[1, 2])

    # m_fac = 1e5 ## TODO

    loss_m = torch.sum(loss_m_inv +
                    loss_m_fwd) / xyz_gt_1.size(0)
    loss_1 = torch.sum(torch.abs(srgb_gt_1 - rend_1) + ( ## TODO maybe change to 0.4 and 0.6
        1.5 * torch.abs(xyz_gt_1 - rec_1))) / xyz_gt_1.size(0)
    loss_2 = torch.sum(torch.abs(srgb_gt_2 - rend_2) + (
        1.5 * torch.abs(xyz_gt_2 - rec_2))) / xyz_gt_2.size(0)

    loss_tot = (loss_1 + loss_2) / 2
    return loss_tot, loss_m

def compute_loss(input, target_xyz, rec_xyz, rendered):
    loss = torch.sum(torch.abs(input - rendered) + (
        1.5 * torch.abs(target_xyz - rec_xyz)))/input.size(0)
    return loss

def from_tensor_to_image(tensor, device='cuda'):
    """ converts tensor to image """
    tensor = torch.squeeze(tensor, dim=0)
    if device == 'cpu':
        image = tensor.data.numpy()
    else:
        image = tensor.cpu().data.numpy()
    # CHW to HWC
    image = image.transpose((1, 2, 0))
    image = from_rgb2bgr(image)
    return image

def from_image_to_tensor(image):
    image = from_bgr2rgb(image)
    image = im2double(image)  # convert to double
    image = np.array(image)
    assert len(image.shape) == 3, ('Input image should be 3 channels colored '
                                   'images')
    # HWC to CHW
    image = image.transpose((2, 0, 1))
    return torch.unsqueeze(torch.from_numpy(image), dim=0)


def from_bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB

def from_rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert from BGR to RGB


def imshow(img, xyz_out=None, srgb_out=None, task=None):
    """ displays images """

    if task.lower() == 'srgb-2-xyz-2-srgb':
        if xyz_out is None:
            raise Exception('XYZ image is not given')
        if srgb_out is None:
            raise Exception('sRGB re-rendered image is not given')

        fig, ax = plt.subplots(1, 3)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('rec. XYZ')
        ax[1].imshow(from_bgr2rgb(xyz_out))
        ax[1].axis('off')
        ax[2].set_title('re-rendered')
        ax[2].imshow(from_bgr2rgb(srgb_out))
        ax[2].axis('off')

    if task.lower() == 'srgb-2-xyz':
        if xyz_out is None:
            raise Exception('XYZ image is not given')

        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('rec. XYZ')
        ax[1].imshow(from_bgr2rgb(xyz_out))
        ax[1].axis('off')

    if task.lower() == 'xyz-2-srgb':
        if srgb_out is None:
            raise Exception('sRGB re-rendered image is not given')

        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('re-rendered')
        ax[1].imshow(from_bgr2rgb(srgb_out))
        ax[1].axis('off')

    if task.lower() == 'pp':
        if srgb_out is None:
            raise Exception('sRGB re-rendered image is not given')

        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('result')
        ax[1].imshow(from_bgr2rgb(srgb_out))
        ax[1].axis('off')

    plt.xticks([]), plt.yticks([])
    plt.show()


def im2double(im):
    """ Returns a double image [0,1] of the uint im. """
    if im[0].dtype == 'uint8':
        max_value = 255
    elif im[0].dtype == 'uint16':
        max_value = 65535
    return im.astype('float') / max_value
