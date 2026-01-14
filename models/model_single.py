from models.hardnet import HarDNet
from models.base import *


class Decoder(nn.Module):
    def __init__(self, full_features, out):
        super(Decoder, self).__init__()
        # self.up1 = UpBlockSkip(full_features[4] + full_features[3], full_features[3],
        #                        func='relu', drop=0).cuda()
        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=0).cuda()
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=0).cuda()
        self.up3 = UpBlockSkip(full_features[1] + full_features[0], full_features[0],
                               func='relu', drop=0).cuda()
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.final = CNNBlock(full_features[0], out, kernel_size=3, drop=0)

    def forward(self, x):
        z = self.up1(x[3], x[2])
        z = self.up2(z, x[1])
        z = self.up3(z, x[0])
        # z = self.up4(z, x[0])
        z = self.Upsample(z)
        out = F.tanh(self.final(z))
        return out


class Unet(nn.Module):
    def __init__(self, order, depth_wise, args):
        super(Unet, self).__init__()
        self.backbone = HarDNet(depth_wise=depth_wise, arch=order, args=args)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = Decoder(d, out=1)
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, img, size=None):
        z = self.backbone(img)
        M = self.decoder(z)
        return M





class SmallDecoder(nn.Module):
    def __init__(self, full_features, out):
        super(SmallDecoder, self).__init__()
        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=0).cuda()
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=0).cuda()
        self.final = CNNBlock(full_features[1], out, kernel_size=3, drop=0)

    def forward(self, x):
        z = self.up1(x[3], x[2])
        z = self.up2(z, x[1])
        # out = torch.tanh(self.final(z))
        out = self.final(z)
        return out


class SparseDecoder(nn.Module):
    def __init__(self, full_features, out, nP, bs):
        super(SparseDecoder, self).__init__()
        self.nP = 64 #nP
        self.embed_dim = 256
        self.cnn = CNNBlock(full_features[-1], 2*self.embed_dim, kernel_size=3, drop=0)
        self.pos_fc = nn.Linear(self.nP*self.embed_dim, self.nP*self.embed_dim)
        self.neg_fc = nn.Linear(self.nP*self.embed_dim, self.nP*self.embed_dim)
        pos_labels = torch.ones(self.nP)
        neg_labels = torch.zeros(self.nP)
        self.labels = torch.cat((pos_labels, neg_labels)).cuda()


    def forward(self, x, embeddings):
        labels = self.labels
        bs = x[-1].shape[0]
        z = self.cnn(x[-1])
        # z = z.reshape(bs, 2*self.embed_dim, -1).permute(0,2,1)
        # z = F.tanh(z)
        # point_embedding = torch.cat([z[... ,:self.embed_dim], z[... ,self.embed_dim:]], dim=1)

        # z = z.reshape(bs, self.embed_dim*self.nP, -1).squeeze()
        z = z.reshape(bs, 2*self.embed_dim, -1).permute(0,2,1)
        z_pos = z[... ,:self.embed_dim]
        z_neg = z[... ,self.embed_dim:]
        
        z_pos = F.relu(z_pos.reshape(bs, self.embed_dim*self.nP, -1).squeeze())
        z_neg = F.relu(z_neg.reshape(bs, self.embed_dim*self.nP, -1).squeeze())

        pos_p = self.pos_fc(z_pos).reshape((bs,self.nP,self.embed_dim))
        neg_p = self.neg_fc(z_neg).reshape((bs,self.nP,self.embed_dim))
        point_embedding = F.tanh(torch.cat([pos_p, neg_p], dim=1))
        
        pad = True
        if pad:
            padding_z = torch.zeros((bs, 1, self.embed_dim), device=z.device)
            padding_label = -torch.ones((1), device=labels.device)
            point_embedding = torch.cat([point_embedding, padding_z], dim=1)
            labels = torch.cat([labels, padding_label]).unsqueeze(0).repeat(bs,1)
        else:
            labels = labels.unsqueeze(0).repeat(bs,1)
        point_embedding[labels == -1] += embeddings[1].weight.clone().detach()
        point_embedding[labels == 0] += embeddings[0][0].weight.clone().detach()
        point_embedding[labels == 1] += embeddings[0][1].weight.clone().detach()
        
        out = point_embedding
        return out
    

class CombDecoder(nn.Module):
    def __init__(self, full_features, out, nP, bs):
        super(CombDecoder, self).__init__()
        self.nP = 64 #nP
        self.embed_dim = 256
        self.cnn = CNNBlock(full_features[-1], 2*self.embed_dim, kernel_size=3, drop=0)
        self.pos_fc = nn.Linear(self.nP*self.embed_dim, self.nP*self.embed_dim)
        self.neg_fc = nn.Linear(self.nP*self.embed_dim, self.nP*self.embed_dim)
        self.drop = nn.Dropout(p = 0.2)
        pos_labels = torch.ones(self.nP)
        neg_labels = torch.zeros(self.nP)
        self.labels = torch.cat((pos_labels, neg_labels)).cuda()
        self.bnorm_pos = nn.BatchNorm1d(self.embed_dim*self.nP)
        self.bnorm_neg = nn.BatchNorm1d(self.embed_dim*self.nP)

        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=0).cuda()
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=0).cuda()
        self.final = CNNBlock(full_features[1], out, kernel_size=3, drop=0)


    def forward(self, x, embeddings):
        labels = self.labels
        bs = x[-1].shape[0]
        z = self.cnn(x[-1])
        # z = z.reshape(bs, 2*self.embed_dim, -1).permute(0,2,1)
        # z = F.tanh(z)
        # point_embedding = torch.cat([z[... ,:self.embed_dim], z[... ,self.embed_dim:]], dim=1)

        # z = z.reshape(bs, self.embed_dim*self.nP, -1).squeeze()
        z = z.reshape(bs, 2*self.embed_dim, -1).permute(0,2,1)
        z_pos = z[... ,:self.embed_dim]
        z_neg = z[... ,self.embed_dim:]
        
        z_pos = F.relu(z_pos.reshape(bs, self.embed_dim*self.nP))
        z_neg = F.relu(z_neg.reshape(bs, self.embed_dim*self.nP))
        
        # z_pos = self.bnorm_pos(z_pos)
        # z_neg = self.bnorm_neg(z_neg)

        pos_p = self.drop(self.pos_fc(z_pos).reshape((bs,self.nP,self.embed_dim)))
        neg_p = self.drop(self.neg_fc(z_neg).reshape((bs,self.nP,self.embed_dim)))
        point_embedding = F.tanh(torch.cat([pos_p, neg_p], dim=1))
        
        pad = True
        if pad:
            padding_z = torch.zeros((bs, 1, self.embed_dim), device=z.device)
            padding_label = -torch.ones((1), device=labels.device)
            point_embedding = torch.cat([point_embedding, padding_z], dim=1)
            labels = torch.cat([labels, padding_label]).unsqueeze(0).repeat(bs,1)
        else:
            labels = labels.unsqueeze(0).repeat(bs,1)
        point_embedding[labels == -1] += embeddings[1].weight.clone().detach()
        point_embedding[labels == 0] += embeddings[0][0].weight.clone().detach()
        point_embedding[labels == 1] += embeddings[0][1].weight.clone().detach()
        
        
        
        z_dense = self.up1(x[3], x[2])
        z_dense = self.up2(z_dense, x[1])
        # z_dense = F.tanh(self.final(z_dense))
        z_dense = self.final(z_dense)

        
        out = (point_embedding, z_dense)
        
        return out


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        nP = int(args['nP']) + 1
        half = 0.5*nP**2
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = Decoder(d, out=4)
        for param in self.backbone.parameters():
            param.requires_grad = True
        x = torch.arange(nP, nP**2, nP).long()
        y = torch.arange(nP, nP**2, nP).long()
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        P = torch.cat((grid_x.unsqueeze(dim=0), grid_y.unsqueeze(dim=0)), dim=0)
        P = P.view(2, -1).permute(1, 0).cuda()
        self.P = (P - half) / half
        pos_labels = torch.ones(P.shape[-2])
        neg_labels = torch.zeros(P.shape[-2])
        self.labels = torch.cat((pos_labels, neg_labels)).cuda().unsqueeze(dim=0)

    def forward(self, img, size=None):
        if size is None:
            half = img.shape[-1] / 2
        else:
            half = size / 2
        P = self.P.unsqueeze(dim=0).repeat(img.shape[0], 1, 1).unsqueeze(dim=1)
        z = self.backbone(img)
        J = self.decoder(z)
        dPx_neg = F.grid_sample(J[:, 0:1], P).transpose(3, 2)
        dPx_pos = F.grid_sample(J[:, 2:3], P).transpose(3, 2)
        dPy_neg = F.grid_sample(J[:, 1:2], P).transpose(3, 2)
        dPy_pos = F.grid_sample(J[:, 3:4], P).transpose(3, 2)
        dP_pos = torch.cat((dPx_pos, dPy_pos), -1)
        dP_neg = torch.cat((dPx_neg, dPy_neg), -1)
        P_pos = dP_pos + P
        P_neg = dP_neg + P
        P_pos = P_pos.clamp(min=-1, max=1)
        P_neg = P_neg.clamp(min=-1, max=1)
        points_norm = torch.cat((P_pos, P_neg), dim=2)
        points = (points_norm * half) + half
        return points, self.labels, J, points_norm


class ModelEmb(nn.Module):
    def __init__(self, args):
        super(ModelEmb, self).__init__()
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = SmallDecoder(d, out=256)
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, img, size=None):
        z = self.backbone(img)
        dense_embeddings = self.decoder(z)
        dense_embeddings = F.interpolate(dense_embeddings, (64, 64), mode='bilinear', align_corners=True)
        return dense_embeddings


class ModelSparseEmb(nn.Module):
    def __init__(self, args):
        super(ModelSparseEmb, self).__init__()
        nP = int(args['nP'])
        bs = int(args['Batch_size'])
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = SparseDecoder(d, out=1, nP=nP, bs=bs)
        for param in self.backbone.parameters():
            param.requires_grad = True
        # pos_labels = torch.ones(int(args['nP']))
        # neg_labels = torch.zeros(int(args['nP']))
        # self.labels = torch.cat((pos_labels, neg_labels)).cuda().unsqueeze(dim=0)

    def forward(self, img, embeddings):
        z = self.backbone(img)
        sparse_embeddings = self.decoder(z, embeddings)
        return sparse_embeddings


class ModelCombEmb(nn.Module):
    def __init__(self, args):
        super(ModelCombEmb, self).__init__()
        nP = int(args['nP'])
        bs = int(args['Batch_size'])
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = CombDecoder(d, out=1, nP=nP, bs=bs)
        for param in self.backbone.parameters():
            param.requires_grad = True
        # pos_labels = torch.ones(int(args['nP']))
        # neg_labels = torch.zeros(int(args['nP']))
        # self.labels = torch.cat((pos_labels, neg_labels)).cuda().unsqueeze(dim=0)

    def forward(self, img, embeddings):
        z = self.backbone(img)
        sparse_embeddings , dense_embeedings = self.decoder(z, embeddings)
        return sparse_embeddings, dense_embeedings

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-nP', '--nP', default=10, help='image size', required=False)
    args = vars(parser.parse_args())

    model = ModelSparseEmb(args=args).cuda()
    x = torch.randn((3, 3, 256, 256)).cuda()
    P = model(x)




