from ref_model import RefineModel
from utils_dino import *
# import smoothness
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
# from datasets.data import SalObjDataset
import torch
import glob
import vision_transformer as vits
import torch.optim as optim
from unwrap_utils import get_tuples, load_input_data
from loss_utils import get_optical_flow_alpha_loss
from tqdm import tqdm
from pathlib import Path
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_h_output(out, video, args):
    folder = os.listdir(args['data_folder'])[video]
    num_of_frame = out.shape[2]
    new_folder = os.path.join(args['data_folder'], folder, 'h_output')
    os.makedirs(new_folder, exist_ok=True)
    for frame in range(num_of_frame):
        curr_map = out[:, :, frame].detach().cpu().numpy()
        curr_path = os.path.join(new_folder, folder + '_' + (5-len(str(frame)))*'0' + str(frame) + '.png')
        cv2.imwrite(curr_path, 255*curr_map)


def evaluate_h_model(video_frames, model_alpha, device, args):
    resx = np.int64(video_frames.shape[1])
    resy = np.int64(video_frames.shape[0])
    larger_dim = np.maximum(resx, resy)
    number_of_frames = video_frames.shape[3]
    alpha_reconstruction = np.zeros((resy, resx, args['c'], number_of_frames))
    with torch.no_grad():
        for f in range(number_of_frames):
            relis_i, reljs_i = torch.where(torch.ones(resy, resx) > 0)
            relisa = np.array_split(relis_i.numpy(), np.ceil(relis_i.shape[0] / 100000))
            reljsa = np.array_split(reljs_i.numpy(), np.ceil(relis_i.shape[0] / 100000))
            for i in range(len(relisa)):
                relis = torch.from_numpy(relisa[i]).unsqueeze(1) / (larger_dim / 2) - 1
                reljs = torch.from_numpy(reljsa[i]).unsqueeze(1) / (larger_dim / 2) - 1
                alpha = model_alpha(torch.cat((reljs, relis, (f / (number_of_frames / 2.0) - 1) *
                                    torch.ones_like(relis)), dim=1).to(device))
                alpha = torch.nn.functional.softmax(alpha,dim=1)
                alpha_reconstruction[relisa[i], reljsa[i], :, f] = alpha.detach().cpu().numpy()
    return alpha_reconstruction
    # return torch.tensor(alpha_reconstruction).cuda().permute(1, 0, 2)


def train_h(h, optimizer, inputs, args):
    criterion = nn.L1Loss()
    size = len(inputs[0])
    larger_dim = np.maximum(args['resx'], args['resy'])
    curr_inx = np.random.randint(0, size)
    video_frames = inputs[1][curr_inx].clone()
    x = video_frames.clone()
    # mask_frames = inputs[3][curr_inx].clone().cuda()
    mask_frames = torch.randint(2, (352, 352, x.shape[-1])).cuda().long()
    number_of_frames = x.shape[3]

    optical_flows_reverse = inputs[5][curr_inx].clone().cuda()
    optical_flows_reverse_mask = inputs[2][curr_inx].clone().cuda()
    optical_flows = inputs[6][curr_inx].clone().cuda()
    optical_flows_mask = inputs[0][curr_inx].clone().cuda()

    # optical_flows_reverse = reshape_flow(optical_flows_reverse, (larger_dim, larger_dim))
    # optical_flows_reverse_mask = reshape_flow(optical_flows_reverse_mask, (larger_dim, larger_dim))
    # optical_flows = reshape_flow(optical_flows, (larger_dim, larger_dim))
    # optical_flows_mask = reshape_flow(optical_flows_mask, (larger_dim, larger_dim))

    jif_all = get_tuples(number_of_frames, x).cuda()
    h.train()
    inds_foreground = torch.randint(jif_all.shape[1],
                                    (np.int64(int(args['samples']) * 1.0), 1)).cuda()

    jif_current = jif_all[:, inds_foreground]  # size (3, batch, 1)
    alpha_maskrcnn = mask_frames[jif_current[1, :], jif_current[0, :],
                                 jif_current[2, :]].squeeze(1).to(device).unsqueeze(-1)
    # alpha_maskrcnn = (alpha_maskrcnn - alpha_maskrcnn.min()) / (alpha_maskrcnn.max() - alpha_maskrcnn.min())
    xyt_current = torch.cat(
        (jif_current[0, :] / (larger_dim / 2) - 1, jif_current[1, :] / (larger_dim / 2) - 1,
         jif_current[2, :] / (number_of_frames / 2.0) - 1),
        dim=1).to(device)  # size (batch, 3)

    # map tanh output of the alpha network to the range (0,1) :
    alpha = h(xyt_current, curr_inx)
    # prevent a situation of alpha=0, or alpha=1 (for the BCE loss that uses log(alpha),log(1-alpha) below)
    # alpha = alpha * 0.99
    # alpha = alpha + 0.001
    # flow_alpha_loss = get_optical_flow_alpha_loss(h, curr_inx,
    #                                               jif_current, alpha, optical_flows_reverse,
    #                                               optical_flows_reverse_mask, larger_dim,
    #                                               number_of_frames, optical_flows,
    #                                               optical_flows_mask, device)
    # alpha_bootstrapping_loss = torch.mean(
    #     -alpha_maskrcnn * torch.log(alpha) - (1 - alpha_maskrcnn) * torch.log(1 - alpha))
    alpha_bootstrapping_loss = criterion(alpha, alpha_maskrcnn)
    loss = alpha_bootstrapping_loss + criterion(alpha.var(), alpha_maskrcnn.var())
    # loss = float(args['w_of'])*flow_alpha_loss + alpha_bootstrapping_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return alpha_bootstrapping_loss


def train_g(model_g, model_org, optimizer, pack, args):
    inputs, _, _, h_maps = pack
    inputs = inputs.cuda().detach()
    h_maps = h_maps.cuda().float().detach()
    Isize = inputs.shape[-1]
    z1 = model(inputs)
    with torch.no_grad():
        z2 = model_org(inputs).detach()
        feats = get_feats_bs(inputs, model_g)
        similars, _ = get_similars_bs(feats, th1=0.0, th2=0.2)
    if args['task'] == 'fine-tune':
        # M = atnn_process(inputs, model, Nmap=4)
        feats = get_feats_bs(inputs, model_g)
        A = (feats @ feats.transpose(1, 2)).squeeze()
        M = get_maps_bs(A, similars, inputs)
        loss = F.mse_loss(norm_batch(M, Isize), norm_batch(h_maps, Isize).detach()) +\
               float(args['w0']) * F.mse_loss(z1, z2.detach())
    else:
        attn, M = atnn_process(inputs, model)
        with torch.no_grad():
            feats = get_feats_bs(inputs, model_g)
            M_u = get_similars_attn(inputs, feats, attn.detach().clone(), th=0.3, a=0)
        loss = F.mse_loss(norm_batch(M, Isize), norm_batch(M_u, Isize).detach()) + \
               float(args['w0']) * F.mse_loss(z1, z2.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def preprocess_h(g, args):
    video_list = open(os.path.join(args['data_folder'], "ImageSets/2017/val.txt")).readlines()
    optical_flows_mask_list = []
    video_frames_list = []
    optical_flows_reverse_mask_list = []
    mask_frames_list = []
    video_frames_dx_list = []
    video_frames_dy_list = []
    optical_flows_reverse_list = []
    optical_flows_list = []
    gt_frames_list = []
    for file in tqdm(video_list[:2]):
        data_folder = Path(args['data_folder'])
        files = os.listdir(args['data_folder'])
        files.sort()
        optical_flows_mask, video_frames, optical_flows_reverse_mask, video_frames_dx, video_frames_dy, \
        optical_flows_reverse, optical_flows = load_input_data(args['resy'], args['resx'],
                                                               args['maximum_number_of_frames'],
                                                               data_folder, file.strip())
        optical_flows_mask_list.append(optical_flows_mask)
        video_frames_list.append(video_frames)
        optical_flows_reverse_mask_list.append(optical_flows_reverse_mask)
        video_frames_dx_list.append(video_frames_dx)
        video_frames_dy_list.append(video_frames_dy)
        optical_flows_reverse_list.append(optical_flows_reverse)
        optical_flows_list.append(optical_flows)
    return optical_flows_mask_list, video_frames_list, optical_flows_reverse_mask_list,\
           video_frames_dx_list, video_frames_dy_list, optical_flows_reverse_list, optical_flows_list


def postprocess_h(h, video_frames, ex):
    out = evaluate_h_model(video_frames, h, device, ex=ex)
    return out


def postprocess_g(g, folder):
    files = os.listdir(folder)
    for file in tqdm(files):
        curr_path = os.path.join(folder, file)
        data_folder = Path(curr_path)
        vid_name = data_folder.name
        preprocess_dino(g, folder, file=vid_name)


def evaluate_g_with_loader(model, root, th=0.75):
    iou_list = []
    trainsize = int(args['resx'])
    img_transform = transforms.Compose([
        transforms.Resize((trainsize, trainsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image_root = os.path.join(root, '*', 'Imgs', '*')
    images = glob.glob(image_root)
    gt_root = os.path.join(root, '*', 'GT_object_level', '*')
    gts = glob.glob(gt_root)
    images = sorted(images)
    gts = sorted(gts)
    for img_path, gt_path in zip(images, gts):
        gt = cv2.imread(gt_path, 0)
        if gt.max() == 0:
            continue
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            image_sizes = (img.height, img.width)
            img_tensor = img_transform(img).unsqueeze(dim=0).cuda()
        res = get_dino_mask(img_tensor, model)[0:1]
        res = (res - res.min()) / (res.max() - res.min())
        res = F.upsample(res, size=image_sizes, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        th_value = res.max() * th
        res[res <= th_value] = 0
        res[res > th_value] = 1
        th_value = gt.max() * 0.5
        gt[gt <= th_value] = 0
        gt[gt > th_value] = 1
        iou = calc_iou(torch.tensor(res), torch.tensor(gt))
        iou_list.append(iou.item())
    return np.mean(iou_list)


def evaluate_model_with_gt(path, data_folder):
    folders = os.listdir(data_folder)
    iou_list = []
    for folder in tqdm(folders):
        res_folder = os.path.join(data_folder, folder, path)
        gt_folder = os.path.join(data_folder, folder, 'GT_object_level')
        if not os.path.exists(res_folder):
            continue
        res_files = os.listdir(res_folder)
        for file in res_files:
            full_path = os.path.join(res_folder, file)
            gt_path = os.path.join(gt_folder, file)
            map = cv2.imread(full_path, 0)
            gt = cv2.imread(gt_path, 0)
            map = cv2.resize(map, (gt.shape[1], gt.shape[0]))
            if gt.max() == 0:
                continue
            th_value = map.max() * 0.5
            map[map <= th_value] = 0
            map[map > th_value] = 1
            th_value = gt.max() * 0.5
            gt[gt <= th_value] = 0
            gt[gt > th_value] = 1
            iou = calc_iou(torch.tensor(map), torch.tensor(gt))
            iou_list.append(iou.item())
    return np.mean(iou_list)


def warmup_h(args=None, h=None, g=None, inputs=None):
    optimizer_h = optim.Adam(h.parameters(),
                             lr=0.0003)
    loss_list = []
    pbar = tqdm(range(int(args['it_h'])))
    if inputs is None:
        inputs = preprocess_h(g, args)
    best = 0
    curr_path = os.path.join('cp',
                             'h_of_' + str(args['w_of']) +
                             '_' + str(datetime.datetime.now()) +
                             '.pth')
    for ep in pbar:
        loss = train_h(h.train(), optimizer_h, inputs, args)
        if len(loss_list) >= 100:
            loss_list = loss_list[1:]
        loss_list.append(loss.item())
        pbar.set_description('loss {bce_loss:.3f}'.format(bce_loss=np.mean(loss_list)))
        if ep % 200 == 199:
            torch.save(h, curr_path)
    return h


def warmup_g(args=None, g=None, org=None):
    optimizer_g = optim.Adam(g.parameters(),
                             lr=float(args['lr_g']),
                             weight_decay=float(args['wd_g']))
    dataset = SalObjDataset(root=args['data_folder'], trainsize=int(args['resx']))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=int(args['bs']),
                                  shuffle=True,
                                  num_workers=int(args['nW']))
    model_path = os.path.join('cp', 'g_' + str(datetime.datetime.now()) + '.pth')
    best = 0
    for ep in range(int(args['it_g'])):
        loss_list = []
        pbar = tqdm(data_loader)
        for ix, pack in enumerate(pbar):
            loss = train_g(g.train(), org.eval(), optimizer_g, pack, args)
            loss_list.append(loss)
            pbar.set_description('loss {bce_loss:.3f}'.format(bce_loss=np.mean(loss_list)))
        iou_g = evaluate_g_with_loader(g, args['test_folder'])
        f.write(str(ep) + ',' +
                str(iou_g) + '\n')
        f.flush()
        if best < iou_g:
            torch.save(g.state_dict(), model_path)
            best = iou_g
    return g


def load_dino(args):
    model = vits.__dict__[args['arch']](patch_size=args['patch_size'], num_classes=0)
    print(f"Model {args['arch']} {args['patch_size']}x{args['patch_size']} built.")
    model.cuda()
    utils.load_pretrained_weights(model, args['pretrained_weights'],
                                  args['checkpoint_key'], args['arch'], args['patch_size'])
    return model


if __name__ == "__main__":
    import argparse
    import datetime

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-it_h', '--it_h', default=100, help='number of workers', required=False)
    parser.add_argument('-it_g', '--it_g', default=1, help='number of workers', required=False)
    parser.add_argument('-lr_g', '--lr_g', default=1e-5, help='number of workers', required=False)
    parser.add_argument('-wd_g', '--wd_g', default=5e-5, help='number of workers', required=False)
    parser.add_argument('-it_ft', '--it_ft', default=10, help='number of workers', required=False)
    parser.add_argument('-patience', '--patience', default=240000, help='number of workers', required=False)
    parser.add_argument('-order', '--order', default=18, help='number of workers', required=False)
    parser.add_argument('-bs', '--bs', default=8, help='number of workers', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='number of workers', required=False)
    parser.add_argument('-maximum_number_of_frames', '--maximum_number_of_frames', default=700, help='dataset task', required=False)
    parser.add_argument('-resx', '--resx', default=352, help='dataset task', required=False)
    parser.add_argument('-resy', '--resy', default=352, help='dataset task', required=False)
    parser.add_argument('-results_folder_name', '--results_folder_name', default='results', help='results', required=False)
    parser.add_argument('-add_to_experiment_folder_name', '--add_to_experiment_folder_name', default='_', help='results', required=False)
    parser.add_argument('-positional_encoding_num_alpha', '--positional_encoding_num_alpha', default=16, help='results', required=False)
    parser.add_argument('-number_of_channels_alpha', '--number_of_channels_alpha', default=1000, help='results', required=False)
    parser.add_argument('-number_of_layers_alpha', '--number_of_layers_alpha', default=8, help='results', required=False)
    parser.add_argument('-samples', '--samples', default=10000, help='results', required=False)
    parser.add_argument('-task', '--task', default='warmup_h', help='results', required=False)
    parser.add_argument('-pre_trained_h', '--pre_trained_h', default='h_of_8_2023-03-03 16:45:52.726335.pth', help='results', required=False)
    parser.add_argument('-pre_trained_g', '--pre_trained_g', default='g_2023-01-10 17:51:21.908190.pth', help='results', required=False)
    parser.add_argument('-data_folder', '--data_folder', default="./data", help='Path to the dataset folder', required=False)
    parser.add_argument('-test_folder', '--test_folder', default="./data", help='Path to the test dataset folder', required=False)
    parser.add_argument('-th', '--th', default=0.5, help='dataset task', required=False)
    parser.add_argument('-w_of', '--w_of', default=8, help='dataset task', required=False)
    parser.add_argument('-w0', '--w0', default=1, help='dataset task', required=False)
    parser.add_argument('-N', '--N', default=1, help='dataset task', required=False)
    parser.add_argument('-th1', '--th1', default=0, help='dataset task', required=False)
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=str, help='')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    args = vars(parser.parse_args())


    cdate = os.path.join('results_dino',
                         args['task'] + '_' +
                         str(datetime.datetime.now()).strip(' ').strip('_').strip('.') + '.csv')
    f = open(cdate, 'w')
    f = save_args_to_csv(f, args)
    f.write('ep,iou_h\n')
    f.flush()

    if args['task'] == 'warmup_h':
        h = RefineModel(args=args, output_dim=2).train().cuda()
        model = load_dino(args)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        warmup_h(args=args, h=h, g=model)
    elif args['task'] == 'warmup_g':
        model = load_dino(args)
        model_org = load_dino(args)
        warmup_g(args=args, g=model, org=model_org)
    elif args['task'] == 'eval_h':
        files = os.listdir(args['data_folder'])
        video_frames_list = []
        files = os.listdir(args['data_folder'])
        files.sort()
        file_h_model = os.path.join('cp', args['pre_trained_h'])
        h = RefineModel(args=args, num_of_ex=50, output_dim=484).cuda().eval()
        h.load_state_dict(torch.load(file_h_model).state_dict())
        for ix, file in enumerate(tqdm(files)):
            curr_path = os.path.join(args['data_folder'], file)
            data_folder = Path(curr_path)
            vid_name = data_folder.name
            vid_root = data_folder.parent
            data_folder = data_folder / 'Imgs'
            input_files = sorted(list(data_folder.glob('*.jpg')) + list(data_folder.glob('*.png')))
            number_of_frames = np.minimum(7000, len(input_files))
            video_frames = torch.zeros((480, 480, 3, number_of_frames))
            for i in range(number_of_frames):
                file1 = input_files[i]
                im = np.array(Image.open(str(file1))).astype(np.float64) / 255.
                video_frames[:, :, :, i] = torch.from_numpy(cv2.resize(im[:, :, :3], (480, 480)))
            out_h = evaluate_h_model(video_frames, h, device, ex=ix)

    elif args['task'] == 'eval_g':
        model = vits.__dict__[args['arch']](patch_size=args['patch_size'], num_classes=0)
        print(f"Model {args['arch']} {args['patch_size']}x{args['patch_size']} built.")
        model.cuda()
        utils.load_pretrained_weights(model, args['pretrained_weights'],
                                      args['checkpoint_key'], args['arch'], args['patch_size'])
        for param in model.parameters():
            param.requires_grad = False
        is_load = args.get('pretrained_weights', '') != ''
        if is_load:
            print('pretrained loaded')
            trained = torch.load(args['pretrained_weights'])
            model.load_state_dict(trained)
        model.eval()
        iou_g = evaluate_g_with_loader(model, args['test_folder'])
        print(iou_g)
    elif args['task'] == 'fine-tune':
        file_h_model = os.path.join('cp', args['pre_trained_h'])
        h = RefineModel(args=args).cuda().eval()
        h.load_state_dict(torch.load(file_h_model).state_dict())
        model = load_dino(args)
        model_org = load_dino(args)
        # inputs = preprocess_h(g, args)
        # for ep in range(int(args['it_ft'])):
        # postprocess_h(h.eval(), inputs, args)
        g = warmup_g(args=args, g=model.train(), org=model_org.eval())
        # inputs = preprocess_h(g, args)
        # h = warmup_h(args=args, h=h.train(), g=g.eval(), inputs=inputs)
