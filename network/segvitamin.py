import torch
from einops import rearrange
from timm.models import VisionTransformer
from torch import nn
from network.vitamin import MbConvStages, VitCfg, VitConvCfg, GeGluMlp, _create_vision_transformer_hybrid



def vitamin_small_diy_size(size=(100, 100), in_chans=3, pretrained=False, **kwargs) -> VisionTransformer:
    stage_1_2 = MbConvStages(cfg=VitCfg(embed_dim=(64, 128, 384),
                                        depths=(2, 4, 1),
                                        stem_width=64,
                                        conv_cfg=VitConvCfg(norm_layer='layernorm2d', norm_eps=1e-6, ),
                                        head_type='1d',),
                             in_chans=in_chans,
                             )
    stage3_args = dict(img_size=size, in_chans = in_chans,  embed_dim=384, depth=14, num_heads=6, mlp_layer=GeGluMlp, mlp_ratio=2., class_token=False, global_pool='avg')
    model = _create_vision_transformer_hybrid('vitamin_small', backbone=stage_1_2, pretrained=pretrained, **dict(stage3_args, **kwargs))
    return model

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups,
                   channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x


class SegVitamin(nn.Module):
    def __init__(self, size=(1024, 768), fact=(4, 4), in_chans=3, out_channel=1):
        super().__init__()
        size2 = (size[0] // fact[0], size[1] // fact[1])
        in_chans2 = in_chans * fact[0] * fact[1]
        out_channel2 = out_channel *fact[0] * fact[1]
        self.fh = fact[0]
        self.fw = fact[1]
        self.vitamin = vitamin_small_diy_size(size=size2, in_chans=in_chans2, num_classes=1).cuda()
        self.features = []
        # 绑定特征层，从vitamin中提取特征
        def get_features(module, input, output):
            self.features.append((input, output))

        for name, module in self.vitamin.named_modules():
            if name.startswith("blocks") and len(name.split('.')) == 2:
                module.register_forward_hook(get_features)
            elif name == "patch_embed.backbone":
                module.register_forward_hook(get_features)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(384 * 6, 192 * 6, kernel_size=2, stride=2, padding=0, groups=6),  # 8
            ChannelShuffle(6),
            nn.BatchNorm2d(192 * 6),
            nn.ReLU(),

            nn.ConvTranspose2d(192 * 6, 96 * 6, kernel_size=2, stride=2, padding=0, groups=6),  # 4
            ChannelShuffle(6),
            nn.Conv2d(96 * 6, 96 * 6, kernel_size=3, stride=1, padding=1, groups=6),  # 4
            nn.BatchNorm2d(96 * 6),
            nn.ReLU(),

            nn.ConvTranspose2d(96 * 6, 48 * 6, kernel_size=2, stride=2, padding=0, groups=6),  # 2
            ChannelShuffle(6),
            nn.BatchNorm2d(48 * 6),
            nn.ReLU(),

            nn.ConvTranspose2d(48 * 6, 24 * 6, kernel_size=2, stride=2, padding=0,  groups=6),  # 1
            nn.BatchNorm2d(24 * 6),
            nn.ReLU(),
            nn.Conv2d(24 * 6, out_channel2, kernel_size=1, stride=1, padding=0, ),
        )

    def forward(self, x):
        x = rearrange(x, "b c (h fh) (w fw) ->b (c fh fw) h w ", fw=self.fw, fh=self.fh)
        self.features = []  # 初始化 features
        # print(x.shape) # w,h   / 4
        self.vitamin(x)  # 执行 vitamin
        features = self.features  # 接受 features
        d_h, d_w = features[0][1].shape[-2:]
        en1 = features[1][1].reshape(-1, 384, d_h, d_w)
        en2 = features[3][1].transpose(-1, -2).reshape(-1, 384, d_h, d_w)
        en3 = features[6][1].transpose(-1, -2).reshape(-1, 384, d_h, d_w)
        en4 = features[9][1].transpose(-1, -2).reshape(-1, 384, d_h, d_w)
        en5 = features[12][1].transpose(-1, -2).reshape(-1, 384, d_h, d_w)
        en6 = features[14][1].transpose(-1, -2).reshape(-1, 384, d_h, d_w)
        y = self.up(torch.cat([en1, en2, en3, en4, en5, en6], dim=1))
        y = rearrange(y, "b (c fh fw) h w -> b c (h fh) (w fw) ", fw=self.fw, fh=self.fh)
        return y


if __name__ == "__main__":
    seg = SegVitamin(size=(1024, 1536), fact=(4, 4), in_chans=1, out_channel=13).cuda()
    # print(seg.vitamin)
    # exit()
    x = torch.rand([2,1, 1024, 1536]).cuda()
    print(x.shape)
    y = seg(x)
    print(y.shape)
    from ptflops import get_model_complexity_info
    res = get_model_complexity_info(seg, (1, 1024, 1536), print_per_layer_stat=True, as_strings=True)
    print(res)
