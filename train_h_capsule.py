from ref_model import RefineModel
from utils_dino import *
import torch
from unwrap_utils import get_tuples, load_input_data
from loss_utils import get_optical_flow_alpha_loss
from tqdm import tqdm
from pathlib import Path
from dino_eval_method import imwrite_indexed, color_palette

import torch.optim as optim
from train_dino_video import evaluate_h_model, save_h_output
from davis2017.davis import DAVIS
import numpy as np
import torch
import cv2
import torch.nn as nn
from urllib.request import urlopen
from davis2017.results import Results
from davis2017 import utils
import os

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dice_coefficient(real_mask, pred_mask):
    """

    :return: dice
    """
    intersection = (real_mask * pred_mask).sum()
    union = real_mask.sum() + pred_mask.sum()

    return 2 * intersection / union, 2 * intersection, union

def get_jaccard_index(real_mask, pred_mask):
    """

    :return: iou
    """
    intersection = (real_mask * pred_mask).sum()
    union = (real_mask | pred_mask).sum()

    return intersection / union

def sort_select(loss, k=0.5, thresh=0.3):
    n = loss.shape[0]
    kn = int(k*n)
    sorted_loss = torch.sort(loss, descending=True)[0]
    if sorted_loss[kn] > thresh:
        loss = sorted_loss[sorted_loss > thresh]
    else:
        loss = sorted_loss[:kn]
    return torch.mean(loss)


def focal_loss(alpha_bootstrapping_loss, loss_flow_alpha_next, loss_flow_alpha_prev, k=0.5):
    return (alpha_bootstrapping_loss ** k).mean(), (loss_flow_alpha_next ** k).mean(), (loss_flow_alpha_prev ** k).mean()


def preprocess(file, args):
    data_folder = Path(args['data_folder'])
    optical_flows_mask, video_frames, optical_flows_reverse_mask, video_frames_dx, video_frames_dy, \
    optical_flows_reverse, optical_flows = load_input_data(args['resy'], args['resx'], args['maximum_number_of_frames'],
                                                           data_folder, file.strip(), args)
    return optical_flows_mask.cuda(), video_frames.cuda(), optical_flows_reverse_mask.cuda(),\
           optical_flows_reverse.cuda(), optical_flows.cuda()


def train_h(h, optimizer, masks, optical_flows_mask, video_frames, optical_flows_reverse_mask, optical_flows_reverse,
            optical_flows, args):
    criterion = nn.CrossEntropyLoss(reduce=False)
    larger_dim = np.maximum(video_frames.shape[1], video_frames.shape[2])
    x = video_frames.clone()
    mask_frames = masks.cuda().permute(1, 2, 0).long()
    number_of_frames = x.shape[3]
    jif_all = get_tuples(number_of_frames, x).cuda()
    inds_foreground = torch.randint(jif_all.shape[1],
                                    (np.int64(int(args['samples']) * 1.0), 1)).cuda()
    jif_current = jif_all[:, inds_foreground]
    alpha_maskrcnn = mask_frames[jif_current[1, :], jif_current[0, :], jif_current[2, :]].squeeze(1).to(device).unsqueeze(-1)
    xyt_current = torch.cat(
        (jif_current[0, :] / (larger_dim / 2) - 1, jif_current[1, :] / (larger_dim / 2) - 1,
         jif_current[2, :] / (number_of_frames / 2.0) - 1),
        dim=1).to(device)  # size (batch, 3)
    alpha = h(xyt_current)
    loss_flow_alpha_next, loss_flow_alpha_prev = get_optical_flow_alpha_loss(h, criterion,
                                                                             jif_current, alpha, optical_flows_reverse,
                                                                             optical_flows_reverse_mask, larger_dim,
                                                                             number_of_frames, optical_flows,
                                                                             optical_flows_mask, device)
    alpha_bootstrapping_loss = criterion(alpha, alpha_maskrcnn.squeeze())
    alpha_bootstrapping_loss = sort_select(alpha_bootstrapping_loss, k=float(args['k']), thresh=float(args['thresh']))
    loss = float(args['w_of'])*loss_flow_alpha_next +\
           float(args['w_of'])*loss_flow_alpha_prev +\
           alpha_bootstrapping_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(alpha_bootstrapping_loss.item(), loss_flow_alpha_next.item(), loss_flow_alpha_prev.item())
    return alpha_bootstrapping_loss


def save_frames(res, video_folder, frame_list):
    for i in range(res.shape[0]):
        frame_nm = frame_list[i].split('/')[-1].replace(".jpg", ".png")
        imwrite_indexed(os.path.join(video_folder, frame_nm), res[i].astype(np.uint8), color_palette)


def evaluate(seq, res_path, dataset_eval, metrics_res):
    metric = ('J', 'F')
    results = Results(root_dir=res_path)
    if args['dataset'] == 'davis2017':
        all_gt_masks, all_void_masks, all_masks_id = dataset_eval.dataset.get_all_masks(video_name.strip(), True)
        all_gt_masks, all_masks_id = all_gt_masks[:, 1:-1, :, :], all_masks_id[1:-1]
        all_res_masks = results.read_masks(seq, all_masks_id)
    else:
        all_gt_masks, all_masks_id = dataset_eval.dataset.get_all_masks_2016(video_name.strip())
        all_res_masks = results.read_masks_2016(seq, all_masks_id)
    j_metrics_res, f_metrics_res = dataset_eval._evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
    for ii in range(all_gt_masks.shape[0]):
        seq_name = f'{seq}_{ii+1}'
        if 'J' in metric:
            [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
            metrics_res['J']["M"].append(JM)
            metrics_res['J']["R"].append(JR)
            metrics_res['J']["D"].append(JD)
            metrics_res['J']["M_per_object"][seq_name] = JM
        if 'F' in metric:
            [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
            metrics_res['F']["M"].append(FM)
            metrics_res['F']["R"].append(FR)
            metrics_res['F']["D"].append(FD)
            metrics_res['F']["M_per_object"][seq_name] = FM
    J = np.mean(list(metrics_res['J']['M_per_object'].values()))
    F = np.mean(list(metrics_res['F']['M_per_object'].values()))
    return J, F


def load_masks(video_name, args, load_gt = False):
    resx = args['resx']
    resy = args['resy']
    if load_gt:
        root_path = os.path.join(args['data_folder'], 'GT')
    else:
        root_path = os.path.join(args['data_folder'], 'preds')
            
    res_path = os.path.join(root_path, video_name)
    files = os.listdir(res_path)
    files.sort()
    omit_files = [file.strip('.png') for file in files]
    results = Results(root_dir=root_path)

    if load_gt:
        new_masks = results.read_segs(video_name, omit_files)
        out = torch.tensor(new_masks).unsqueeze(dim=1).cuda()
        return out.long().squeeze(), files
    else:
        new_masks = results.read_segs(video_name, omit_files)[...,0]
        out = torch.tensor(new_masks).unsqueeze(dim=1).cuda()
        return F.interpolate(out, (resy, resx), mode='nearest').long().squeeze(), files


def open_results_folder(args):
    cfolder = os.path.join('results_h', 'test_time_' + args['dataset'],
                           str(datetime.datetime.now()).strip(' ').strip('_').strip('.'))
    os.makedirs(cfolder, exist_ok=True)
    csv_file = os.path.join(cfolder, 'results.csv')
    our_folder = os.path.join(cfolder, 'ours')
    dino_folder = os.path.join(cfolder, 'dino')
    os.makedirs(our_folder, exist_ok=True)
    os.makedirs(dino_folder, exist_ok=True)
    f = open(csv_file, 'w')
    f = save_args_to_csv(f, args)
    f.write('video, dice_ours, iou_ours, dice_sam, iou_sam\n')
    f.flush()
    return f, our_folder, dino_folder


def get_metrics():
    metric = ('J', 'F')
    metrics_res_ours = {}
    # metrics_res_dino = {}
    if 'J' in metric:
        metrics_res_ours['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    if 'F' in metric:
        metrics_res_ours['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    return metrics_res_ours


def get_paths(args):
    """Get video list based on dataset type. Paths should be provided via command line arguments."""
    args['data_path'] = args['data_folder']
    if args['dataset'] == 'davis2017':
        video_list = open(os.path.join(args['data_path'], "ImageSets/2017/val.txt")).readlines()
    elif args['dataset'] in ['davis2016', 'fbms59', 'segtrackv2']:
        video_list = os.listdir(os.path.join(args['data_folder'], 'JPEGImages'))
    elif args['dataset'] in ['sun-easy', 'sun-hard', 'capsule']:
        video_list = os.listdir(os.path.join(args['data_folder'], 'Frame'))
    else:
        raise ValueError(f"Unknown dataset: {args['dataset']}")
    return args, video_list


if __name__ == "__main__":
    import argparse
    import datetime
    from davis2017.evaluation import DAVISEvaluation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-it', '--it', default=500, help='number of workers', required=False)
    parser.add_argument('-lr', '--lr', default=1e-3, help='number of workers', required=False)
    parser.add_argument('-wd', '--wd', default=5e-5, help='number of workers', required=False)
    parser.add_argument('-resx', '--resx', default=336, help='number of workers', required=False)
    parser.add_argument('-resy', '--resy', default=336, help='number of workers', required=False)
    parser.add_argument('-maximum_number_of_frames', '--maximum_number_of_frames', default=700, help='dataset task', required=False)
    parser.add_argument('-add_to_experiment_folder_name', '--add_to_experiment_folder_name', default='_', help='results', required=False)
    parser.add_argument('-positional_encoding_num_alpha', '--positional_encoding_num_alpha', default=16, help='results', required=False)
    parser.add_argument('-number_of_channels_alpha', '--number_of_channels_alpha', default=400, help='results', required=False)
    parser.add_argument('-number_of_layers_alpha', '--number_of_layers_alpha', default=8, help='results', required=False)
    parser.add_argument('-samples', '--samples', default=50000, help='results', required=False)
    parser.add_argument('-data_folder', '--data_folder', default="./data", help='Path to the dataset folder', required=False)
    parser.add_argument('-data_path', '--data_path', default="./data", help='Path to the dataset folder', required=False)
    parser.add_argument('-w_of', '--w_of', default=1, help='dataset task', required=False)
    parser.add_argument('-k', '--k', default=0.99, help='dataset task', required=False)
    parser.add_argument('-thresh', '--thresh', default=0.1, help='dataset task', required=False)
    parser.add_argument('--arch', default='vit_base', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=str, help='')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--dataset', default='capsule', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    args = vars(parser.parse_args())

    f, our_folder, dino_folder = open_results_folder(args)
    args, video_list = get_paths(args)

    args['resx'] = int(args['resx'])
    args['resy'] = int(args['resy'])
    metrics_res_ours, metrics_res_dino = get_metrics()
    video_name = "B"
    our_video = os.path.join(our_folder, video_name.strip())
    os.makedirs(our_video, exist_ok=True)
    dino_video = os.path.join(dino_folder, video_name.strip())
    os.makedirs(dino_video, exist_ok=True)
    with torch.no_grad():
        masks, frame_list = load_masks(video_name.strip(), args)
        masks[masks<=128] = 0
        masks[masks>128] = 1
        args['resx'] = masks.shape[2]
        args['resy'] = masks.shape[1]
    if args['dataset'] == 'davis2017':
        c = len(np.unique(masks.detach().cpu()))
    else:
        c = 2
    args['c'] = c
    optical_flows_mask, video_frames, optical_flows_reverse_mask, optical_flows_reverse, optical_flows = \
        preprocess(video_name, args)
    h = RefineModel(args=args, output_dim=c).train().cuda()
    optimizer = optim.Adam(h.parameters(),
                            lr=float(args['lr']),
                            weight_decay=float(args['wd']))
    pbar = tqdm(range(int(args['it'])))
    print(video_name)
    loss_list = []
    for it in pbar:
        loss = train_h(h, optimizer, masks, optical_flows_mask, video_frames, optical_flows_reverse_mask,
                        optical_flows_reverse, optical_flows, args)
        loss_list.append(loss.item())
        pbar.set_description('loss {bce_loss:.3f}'.format(bce_loss=np.mean(loss_list)))
    with torch.no_grad():
        all_gt_masks, gt_frame_list = load_masks(video_name.strip(), args, load_gt=True)
        all_gt_masks[all_gt_masks<=128] = 0
        all_gt_masks[all_gt_masks>128] = 1
        size = all_gt_masks.shape[1:]
        out = evaluate_h_model(video_frames, h.eval(), device, args)
        out = torch.tensor(out[:,:,1,:]).permute(2, 0, 1).unsqueeze(dim=1)
        out = F.interpolate(out.float(), size, mode='bilinear').squeeze().numpy()
        out_th = out>0.5
        masks = F.interpolate(masks.cpu().unsqueeze(dim=1).float(), size, mode='nearest').squeeze().numpy()
        save_frames(masks*255, dino_video, frame_list)
        save_frames(out*255, our_video, frame_list)
        dice_ours,_,_ = get_dice_coefficient(all_gt_masks.cpu().numpy(),out_th)
        iou_ours = get_jaccard_index(all_gt_masks.cpu().numpy().astype(bool),out_th.astype(bool))
        dice_sam,_,_ = get_dice_coefficient(all_gt_masks.cpu().numpy(),masks)
        iou_sam = get_jaccard_index(all_gt_masks.cpu().numpy().astype(bool),masks.astype(bool))
        f.write(video_name.strip() + ',' +
                str(dice_ours)[:6] + ',' +
                str(iou_ours)[:6] + ',' +
                str(dice_sam)[:6] + ',' +
                str(iou_sam)[:6] + ',' +
                str(dice_ours - dice_sam)[:6] + ',' +
                str(iou_ours - iou_sam)[:6] + ',' + '\n')
        f.flush()
        print(f'dice: {dice_ours:.3f}, iou: {iou_ours:.3f}, dice_sam: {dice_sam:.3f}, iou_sam: {iou_sam:.3f}')


