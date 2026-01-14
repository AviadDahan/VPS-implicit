from davis2017.evaluation import DAVISEvaluation
import numpy as np
import utils
import vision_transformer as vits
import os
import glob
import torch
from PIL import Image
import queue
import cv2
from tqdm import tqdm
from torch.nn import functional as F
import copy
from urllib.request import urlopen


color_palette = []
for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
    color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1, 3)


def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if(n_dims is None):
        n_dims = int(y_tensor.max()+ 1)
    _, h, w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)


def read_seg(seg_dir, factor, scale_size=[480]):
    seg = Image.open(seg_dir)
    _w, _h = seg.size # note PIL.Image.Image's size is (w, h)
    if len(scale_size) == 1:
        if(_w > _h):
            _th = scale_size[0]
            _tw = (_th * _w) / _h
            _tw = int((_tw // 64) * 64)
        else:
            _tw = scale_size[0]
            _th = (_tw * _h) / _w
            _th = int((_th // 64) * 64)
    else:
        _th = scale_size[1]
        _tw = scale_size[0]
    small_seg = np.array(seg.resize((_tw // factor, _th // factor), 0))
    small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
    return to_one_hot(small_seg), np.asarray(seg)


def read_frame_list(video_dir, sufix="*.jpg"):
    frame_list = [img for img in glob.glob(os.path.join(video_dir, sufix))]
    frame_list = sorted(frame_list)
    return frame_list


def extract_feature(model, frame, return_h_w=False):
    """Extract one frame feature everytime."""
    out = model.get_intermediate_layers(frame.unsqueeze(0).cuda(), n=1)[0]
    out = out[:, 1:, :]  # we discard the [CLS] token
    h, w = int(frame.shape[1] / model.patch_embed.patch_size), int(frame.shape[2] / model.patch_embed.patch_size)
    dim = out.shape[-1]
    out = out[0].reshape(h, w, dim)
    out = out.reshape(-1, dim)
    if return_h_w:
        return out, h, w
    return out


def label_propagation(args, model, frame_tar, list_frame_feats, list_segs, mask_neighborhood=None):
    """
    propagate segs of frames in list_frames to frame_tar
    """
    ## we only need to extract feature of the target frame
    feat_tar, h, w = extract_feature(model, frame_tar, return_h_w=True)

    return_feat_tar = feat_tar.T # dim x h*w

    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w

    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1) # nmb_context x h*w (tar: query) x h*w (source: keys)

    if mask_neighborhood is None:
        mask_neighborhood = restrict_neighborhood(h, w)
        mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
    aff *= mask_neighborhood

    aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
    tk_val, _ = torch.topk(aff, dim=0, k=5)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)
    return seg_tar, return_feat_tar, mask_neighborhood


def restrict_neighborhood(h, w):
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    size_mask_neighborhood = 12
    for i in range(h):
        for j in range(w):
            for p in range(2 * size_mask_neighborhood + 1):
                for q in range(2 * size_mask_neighborhood + 1):
                    if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                        continue
                    if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)


def read_frame(frame_dir, scale_size=[480]):
    """
    read a single frame & preprocess
    """
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if(ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 64) * 64)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 64) * 64)
        img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    return img, ori_h, ori_w


def eval_devis(davis_path, task, subset, results_path):
    print(f'Evaluating sequences for the {task} task...')
    dataset_eval = DAVISEvaluation(davis_root=davis_path, task=task, gt_set=subset)
    metrics_res = dataset_eval.evaluate(results_path)
    J = np.mean(list(metrics_res['J']['M_per_object'].values()))
    F = np.mean(list(metrics_res['F']['M_per_object'].values()))
    final_mean = (J + F) / 2.
    return final_mean, J, F


def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt,:,:] = mask_cnt
    return mask


def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png for DAVIS."""
    # if np.atleast_3d(array).shape[2] != 1:
    #   raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    # im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')


def eval_devis_loader(model, args):
    video_list = open(os.path.join(args.data_path, "ImageSets/2017/val.txt")).readlines()
    for i, video_name in enumerate(video_list):
        video_name = video_name.strip()
        video_dir = os.path.join(args.data_path, "JPEGImages/480p/", video_name)
        video_folder = os.path.join(args.output_dir, video_dir.split('/')[-1])
        os.makedirs(video_folder, exist_ok=True)
        print(f'[{i}/{len(video_list)}] Begin to segmentate video {video_name}.')
        video_dir = os.path.join(args.data_path, "JPEGImages/480p/", video_name)
        frame_list = read_frame_list(video_dir)
        seg_path = frame_list[0].replace("JPEGImages", "Annotations").replace("jpg", "png")
        first_seg, seg_ori = read_seg(seg_path, args.patch_size)
        que = queue.Queue(args.n_last_frames)

        frame1, ori_h, ori_w = read_frame(frame_list[0])
        frame1_feat = extract_feature(model, frame1).T  # dim x h*w
        mask_neighborhood = None
        for cnt in tqdm(range(1, len(frame_list))):
            frame_tar = read_frame(frame_list[cnt])[0]
            used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
            used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]
            frame_tar_avg, feat_tar, mask_neighborhood = label_propagation(args, model, frame_tar, used_frame_feats,
                                                                           used_segs, mask_neighborhood)
            # pop out oldest frame if neccessary
            if que.qsize() == args.n_last_frames:
                que.get()
            # push current results into queue
            seg = copy.deepcopy(frame_tar_avg)
            que.put([feat_tar, seg])
            # upsampling & argmax
            frame_tar_avg = \
            F.interpolate(frame_tar_avg, scale_factor=args.patch_size, mode='bilinear', align_corners=False,
                          recompute_scale_factor=False)[0]
            frame_tar_avg = norm_mask(frame_tar_avg)
            _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

            # saving to disk
            frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
            frame_tar_seg = np.array(Image.fromarray(frame_tar_seg).resize((ori_w, ori_h), 0))
            frame_nm = frame_list[cnt].split('/')[-1].replace(".jpg", ".png")
            imwrite_indexed(os.path.join(video_folder, frame_nm), frame_tar_seg, color_palette)
    return eval_devis(davis_path=args.data_path, task='semi-supervised', subset='val', results_path='vis/')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Evaluation with video object segmentation on DAVIS 2017')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--output_dir', default="vis/", help='Path where to save segmentations')
    parser.add_argument('--data_path', default='./data', type=str, help='Path to the dataset')
    parser.add_argument("--n_last_frames", type=int, default=7, help="number of preceeding frames")
    parser.add_argument("--size_mask_neighborhood", default=12, type=int,
                        help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--topk", type=int, default=5, help="accumulate label from top k neighbors")
    parser.add_argument("--bs", type=int, default=6, help="Batch size, try to reduce if OOM")
    args = parser.parse_args()

    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    print(eval_devis(davis_path=args.data_path,
                     task='semi-supervised',
                     subset='val',
                     results_path='vis/'
                     )
          )

    # print(
    #     eval_devis_loader(davis_path=args.data_path, task='val', model=model, args=args)
    # )

