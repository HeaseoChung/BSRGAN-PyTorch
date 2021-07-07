import argparse
from os import terminal_size

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import RRDBNet
from utils import preprocess, get_concat_h

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = RRDBNet(scale_factor=args.scale).to(device)
    try:
        model.load_state_dict(torch.load(args.weights_file, map_location=device))
    except:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file,map_location=device).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
    
    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    lr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    lr = preprocess(lr).to(device)
    bic = preprocess(bicubic).to(device)

    with torch.no_grad():
        preds = model(lr)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0)

    output = np.array(preds).transpose([1,2,0])
    output = np.clip(output, 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_BSRGAN_x{}.'.format(args.scale)))

    if args.merge:
        merge = get_concat_h(bicubic, output).save(args.image_file.replace('.', '_hconcat_.'))