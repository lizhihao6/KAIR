import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import cv2
import numpy as np


class DatasetTISR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetTISR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 1
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        if 'dataroot_H_val' in opt.keys():
            self.paths_H += util.get_image_paths(opt['dataroot_H_val'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image & L image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = cv2.imread(H_path, 0) # gray scale

        # resize
        h, w = img_H.shape[:2]
        noise = np.random.normal(0, 10**0.5, img_H.shape)
        img_L = img_H + noise
        cv2.normalize(img_L, img_L, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        img_L = img_L.astype(np.uint8)
        img_L = cv2.resize(img_L, (w//self.sf, h//self.sf), interpolation=cv2.INTER_CUBIC)

        # jpeg comp
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, encimg = cv2.imencode('.jpg', img_L, encode_param)
        img_L = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)


        if len(img_H.shape) == 2:
            assert len(img_L.shape) == 2
            img_H = np.expand_dims(img_H, axis=2)
            img_L = np.expand_dims(img_L, axis=2)
        img_H, img_L = img_H.astype(np.float32)/255, img_L.astype(np.float32)/255 

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        # gray to three channels
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        img_H, img_L = img_H.repeat([3, 1, 1]), img_L.repeat([3, 1, 1])

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
