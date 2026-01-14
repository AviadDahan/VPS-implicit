import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def preprocess_dino(model, data_folder, file):
    folder = os.path.join(data_folder, file, 'Imgs')
    new_folder = os.path.join(data_folder, file, 'dino')
    os.makedirs(new_folder, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dino_masks = generate_dino_affinity(folder, transform, model)
    return dino_masks


def generate_dino_affinity(folder, transform, model):
    files = os.listdir(folder)
    model.eval()
    out = []
    for file in files:
        path = os.path.join(folder, file)
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            img = transform(img).repeat(1, 1, 1, 1).cuda()
            res = get_dino_affinity(img, model)
            # res = (res - res.min()) / (res.max() - res.min())
            out.append(res.unsqueeze(dim=0))
            # for i in range(N):
            #     tmp = (res[i] - res[i].min()) / (res[i].max() - res[i].min() + 1e-8)
            #     tmp = (255 * tmp.squeeze().cpu().numpy()).astype(np.uint8)
            #     if i == 0:
            #         outfile = os.path.join(new_folder, file[:-4] + '.png')
            #     else:
            #         outfile = os.path.join(new_folder, file[:-4] + '_' + str(i) + '.png')
            #     cv2.imwrite(outfile, tmp)
    return torch.cat(out, dim=0).permute(1, 2, 0).unsqueeze(dim=1)


def get_dino_affinity(img, model):
    feats = get_feats_bs(img, model)
    A = (feats @ feats.transpose(1, 2)).squeeze()
    return A

def generate_dino_masks(new_folder, folder, transform, model, N, th1=0, th2=0.2):
    files = os.listdir(folder)
    model.eval()
    out = []
    for file in files:
        path = os.path.join(folder, file)
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            img = transform(img).repeat(1, 1, 1, 1).cuda()
            res = get_dino_mask(img, model, N, th1, th2)
            res = (res - res.min()) / (res.max() - res.min())
            th_value = res.max() * 0.75
            res[res <= th_value] = 0
            res[res > th_value] = 1
            out.append(res)
            for i in range(N):
                tmp = (res[i] - res[i].min()) / (res[i].max() - res[i].min() + 1e-8)
                tmp = (255 * tmp.squeeze().cpu().numpy()).astype(np.uint8)
                if i == 0:
                    outfile = os.path.join(new_folder, file[:-4] + '.png')
                else:
                    outfile = os.path.join(new_folder, file[:-4] + '_' + str(i) + '.png')
                cv2.imwrite(outfile, tmp)
    return torch.cat(out, dim=1).permute(2, 3, 0, 1)


def calc_iou(alpha_reconstruction, gt_frames):
    intersection = (alpha_reconstruction*gt_frames).sum(dim=(1, 0))
    union = alpha_reconstruction.sum(dim=(1, 0)) + gt_frames.sum(dim=(1, 0)) - intersection
    iou = intersection / union
    mean_iou = iou.mean()
    return mean_iou


def save_args_to_csv(f, args):
    for item in args.keys():
        value = args[item]
        f.write(str(item) + ',' + str(value) + '\n')
        f.flush()
    return f


def get_dino_mask(img, model, N=2, th1=0, th2=0.2):
    feats = get_feats_bs(img, model)
    similars, A = get_similars_single(feats=feats, N=N, th1=th1, th2=th2)
    out = []
    for i in range(N):
        M_u = get_maps(A, similars, i, img)
        out.append(M_u)
    return torch.cat(out)


def get_feats_bs(img, model):
    # Store the outputs of qkv layer from the last attention layer
    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
    # Forward pass in the model
    attentions = model.get_last_selfattention(img)
    # Dimensions
    nb_im = attentions.shape[0]  # Batch size
    nh = attentions.shape[1]  # Number of heads
    nb_tokens = attentions.shape[2]  # Number of tokens
    qkv = (
        feat_out["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4)
    )
    _, k, _ = qkv[0], qkv[1], qkv[2]
    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    feats = k[:, 1:, :]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def get_maps(A, similars, i, real_imgs):
    if similars[i].shape[0] == 1:
        inx = similars[i].squeeze().item()
        M = A[inx, :]
    else:
        M = torch.mean(A[similars[i].squeeze(), :], dim=0, keepdim=True).squeeze()
    w_featmap, h_featmap = real_imgs.shape[2] // 16, real_imgs.shape[3] // 16
    M = M.reshape(w_featmap, h_featmap).float().unsqueeze(dim=0).unsqueeze(dim=0)
    M = F.interpolate(M, real_imgs.shape[2:], mode='bilinear', align_corners=True)
    return M


def get_maps_bs(A, similars, real_imgs):
    bs = A.shape[0]
    M = torch.zeros((bs, A.shape[2])).cuda()
    for i in range(bs):
        tmp = torch.mean(A[i:i + 1, similars[i], :], dim=1)
        M[i] = tmp
    w_featmap, h_featmap = real_imgs.shape[2] // 16, real_imgs.shape[3] // 16
    M = M.reshape(bs, w_featmap, h_featmap).float().unsqueeze(dim=1)
    M = F.interpolate(M, real_imgs.shape[2:], mode='bilinear', align_corners=True)
    # M = norm_batch(M, real_imgs.shape[2])
    return M


def get_similars_single(feats, N=1, th1=0, th2=0.2):
    A = (feats @ feats.transpose(1, 2)).squeeze()
    out = []
    tmp = A.detach().clone()
    for i in range(N):
        sorted_patches, scores = patch_scoring(tmp)
        seed = sorted_patches[0]
        similars = get_similars(tmp, seed, th1=th1, th2=th2)
        out.append(similars)
        tmp[similars.squeeze(), :] = -1
        tmp[:, similars.squeeze()] = -1
    return out, A


def get_similars_bs(feats, th1=0.0, th2=0.2):
    bs = feats.shape[0]
    M = (feats @ feats.transpose(1, 2)).squeeze()
    out = []
    for i in range(bs):
        A = M[i]
        sorted_patches, scores = patch_scoring(A)
        seed = sorted_patches[0]
        similars = get_similars(A, seed, th1=th1, th2=th2)
        out.append(similars)
    return out, M


def patch_scoring(M, threshold=0.):
    """
    Patch scoring based on the inverse degree.
    """
    # Cloning important
    A = M.clone()

    # Zero diagonal
    A.fill_diagonal_(0)

    # Make sure symmetric and non nul
    A[A < 0] = 0
    C = A + A.t()

    # Sort pixels by inverse degree
    cent = -torch.sum(A > threshold, dim=1).type(torch.float32)
    cent[cent == 0] = -1000000
    sel = torch.argsort(cent, descending=True)

    return sel, cent


def get_similars(A, seed, th1=0, th2=0.2):
    M = A.clone()
    M[M >= th1] = 1
    M[M < th1] = 0
    curr_iou = calc_iou_dino(M[seed:seed+1], M)
    return torch.nonzero(curr_iou > th2)


def calc_iou_dino(x, y):
    intersection = (x*y).sum(dim=1)
    union = x.sum(dim=1) + y.sum(dim=1) - intersection
    iou = intersection / union
    return iou


def atnn_process(real_imgs, model, Nmap=None):
    bs = real_imgs.shape[0]
    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
    attn = model.get_last_selfattention(real_imgs)
    w_featmap, h_featmap = real_imgs.shape[3] // 16, real_imgs.shape[3] // 16
    nh = attn.shape[1]
    attentions = attn[:, :, 0, 1:].reshape(bs, nh, -1)
    th_attn = attentions.reshape(bs, nh, w_featmap, h_featmap).float()
    if Nmap == None:
        return attentions, F.interpolate(th_attn, real_imgs.shape[2:], mode='bilinear', align_corners=True)
    return F.interpolate(th_attn[:, Nmap:Nmap+1, :, :], real_imgs.shape[2:], mode='bilinear', align_corners=True)


def norm_batch(x, Isize):
    bs = x.shape[0]
    min_value = x.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    max_value = x.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x


def get_similars_attn(real_imgs, feats, attn, th, a):
    M = (feats @ feats.transpose(1, 2)).squeeze()
    attn_u = torch.zeros(attn.shape).float().cuda()
    w_featmap, h_featmap = real_imgs.shape[3] // 16, real_imgs.shape[3] // 16
    nh = attn.shape[1]
    for bs in range(M.shape[0]):
        for i in range(nh):
            tmp = norm(attn[bs:bs+1, i, :])
            curr_iou = calc_iou_dino(thresholding(M[bs].clone(), 0.5), thresholding(tmp.clone(), 0.5))
            similars = torch.nonzero(curr_iou > th)
            if similars.shape[0] > a:
                attn_u[bs, i] = torch.mean(M[bs][similars.squeeze()], dim=0)
            else:
                attn_u[bs, i] = attn[bs, i]
    attn_u = attn_u.reshape(M.shape[0], nh, w_featmap, h_featmap).float()
    attn_u = F.interpolate(attn_u, real_imgs.shape[2:], mode='bilinear', align_corners=True)
    return attn_u


def thresholding(M, th=0):
    M[M >= th] = 1
    M[M < th] = 0
    return M


def norm(x):
    return (x - x.min()) / (x.max() - x.min())


def reshape_flow(flow_org, size):
    flow = flow_org.clone()
    if len(flow.shape) == 5:
        flow = flow.squeeze().permute(3, 2, 0, 1)
        flow = F.interpolate(flow, size=size, mode='bilinear', align_corners=True)
        return flow.permute(2, 3, 1, 0).unsqueeze(dim=4)
    elif flow.shape[2] == 3:
        flow = flow.permute(3, 2, 0, 1)
        flow = F.interpolate(flow, size=size, mode='bilinear', align_corners=True)
        return flow.permute(2, 3, 1, 0)
    else:
        flow = flow.permute(2, 3, 0, 1)
        flow = F.interpolate(flow, size=size, mode='bilinear', align_corners=True)
        return flow.permute(2, 3, 0, 1)

