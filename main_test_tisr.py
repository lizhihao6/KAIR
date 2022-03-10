import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
from pathlib import Path

from models.network_swinir import SwinIR as net
from utils import utils_image as util

TISR_PATH = '/shared/PBVS2022/TISR/test/evaluation1/hr_x4/'
TISR_MR_PATH = '/shared/PBVS2022/TISR/test/evaluation2/mr_real/'


def augment_img_tensor4(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return img.rot90(1, [2, 3]).flip([2])
    elif mode == 2:
        return img.flip([2])
    elif mode == 3:
        return img.rot90(3, [2, 3])
    elif mode == 4:
        return img.rot90(2, [2, 3]).flip([2])
    elif mode == 5:
        return img.rot90(1, [2, 3])
    elif mode == 6:
        return img.rot90(2, [2, 3])
    elif mode == 7:
        return img.rot90(3, [2, 3]).flip([2])

def inv_augment_img_tensor4(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return img.flip([2]).rot90(3, [2, 3])
    elif mode == 2:
        return img.flip([2])
    elif mode == 3:
        return img.rot90(1, [2, 3])
    elif mode == 4:
        return img.flip([2]).rot90(2, [2, 3])
    elif mode == 5:
        return img.rot90(3, [2, 3])
    elif mode == 6:
        return img.rot90(2, [2, 3])
    elif mode == 7:
        return img.flip([2]).rot90(1, [2, 3])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='tisr', help='task name')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--training_patch_size', type=int, default=64, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='superresolution/swinir_sr_tisr_patch64_x4/models/7000_G.pth')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--package', action='store_true', help='zip the package')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        raise NotImplementedError
        
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)
    
    if args.package:
        package(save_dir)
    

def package(save_dir):
    os.system('rm submitStructure.zip')
    os.system('cp -r /shared/PBVS2022/TISR/submitStructure ./')
    pngs = [str(s) for s in Path(save_dir).glob('*.png')]
    for p in pngs:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        imgname = os.path.basename(p).split('_')[1][1:]
        cv2.imwrite('submitStructure/evaluation1/x4/' + 'ev1_' + imgname + '.png', im)
    
    jpgs = [str(s) for s in Path(TISR_MR_PATH).glob('*.jpg')]
    for p in jpgs:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        im = cv2.resize(im, (w*2, h*2))
        imgname = os.path.basename(p).split('.')[0][1:]
        cv2.imwrite('submitStructure/evaluation2/x2/' + 'ev2_' + imgname + '.png', im)

    os.system('cd submitStructure && zip -r ../submitStructure.zip ./* && cd ../')
    os.system('rm -r submitStructure')


def define_model(args):
    if args.task == 'tisr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'
    else:
        raise NotImplementedError
    
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model


def setup(args):
    if args.task == 'tisr':
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        folder = TISR_PATH
        border = args.scale
        window_size = 8
    else:
        raise NotImplementedError

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    if args.task == 'tisr':
        img_gt = None
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    else:
        raise NotImplementedError

    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq.clone())
        for i in range(1, 8):
            _img_lq = augment_img_tensor4(img_lq.clone(), mode=i)
            _output = model(_img_lq)
            _output = inv_augment_img_tensor4(_output, mode=i)
            output += _output
        output /= 8
    else:
        raise NotImplementedError

    return output

if __name__ == '__main__':
    main()
