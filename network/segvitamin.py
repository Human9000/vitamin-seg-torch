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
                                        head_type='1d', ),
                             in_chans=in_chans,
                             )
    stage3_args = dict(img_size=size, in_chans=in_chans, embed_dim=384, depth=14, num_heads=6, mlp_layer=GeGluMlp, mlp_ratio=2., class_token=False,
                       global_pool='avg')
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
        out_channel2 = out_channel * fact[0] * fact[1]
        self.fh = fact[0]
        self.fw = fact[1]
        self.vitamin = vitamin_small_diy_size(size=size2, in_chans=in_chans2, num_classes=1).cuda()
        self.features = []
        self.dw = -1
        self.dh = -1

        # 绑定特征层，从vitamin中提取特征
        def get_features(module, input, output):
            self.features.append(output)

        def get_dw_dh(module, input, output):
            self.dh = output.shape[-2]
            self.dw = output.shape[-1]

        for name, module in self.vitamin.named_modules():
            if name.startswith("blocks") and len(name.split('.')) == 2:
                module.register_forward_hook(get_features)
            elif name == "patch_embed.backbone":
                module.register_forward_hook(get_dw_dh)

        group = 6
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(384 * group, 192 * group, kernel_size=2, stride=2, padding=0, groups=group),  # 8
            ChannelShuffle(6),
            nn.BatchNorm2d(192 * 6),
            nn.ReLU(),

            nn.ConvTranspose2d(192 * group, 96 * group, kernel_size=2, stride=2, padding=0, groups=group),  # 4
            ChannelShuffle(6),
            nn.Conv2d(96 * group, 96 * group, kernel_size=3, stride=1, padding=1, groups=group),  # 4
            nn.BatchNorm2d(96 * group),
            nn.ReLU(),

            nn.ConvTranspose2d(96 * group, 48 * group, kernel_size=2, stride=2, padding=0, groups=group),  # 2
            ChannelShuffle(6),
            nn.BatchNorm2d(48 * group),
            nn.ReLU(),

            nn.ConvTranspose2d(48 * group, 24 * group, kernel_size=2, stride=2, padding=0, groups=group),  # 1
            nn.BatchNorm2d(24 * group),
            nn.ReLU(),
            nn.Conv2d(24 * group, out_channel2, kernel_size=1, stride=1, padding=0, ),
        )

    def encode(self, x):
        self.features = []  # 初始化 features
        self.vitamin(x)  # 执行 vitamin，将特征保存在 features 中
        # 提取特征层
        en0 = self.features[0 ].transpose(-1, -2).reshape(-1, 384, self.dh, self.dw)
        en1 = self.features[3 ].transpose(-1, -2).reshape(-1, 384, self.dh, self.dw)
        en2 = self.features[6 ].transpose(-1, -2).reshape(-1, 384, self.dh, self.dw)
        en3 = self.features[9 ].transpose(-1, -2).reshape(-1, 384, self.dh, self.dw)
        en4 = self.features[11].transpose(-1, -2).reshape(-1, 384, self.dh, self.dw)
        en5 = self.features[13].transpose(-1, -2).reshape(-1, 384, self.dh, self.dw)
        en = torch.cat([en0, en1, en2, en3, en4, en5], dim=1)
        return en

    def forward(self, x):
        x = rearrange(x, "b c (h fh) (w fw) ->b (c fh fw) h w ", fw=self.fw, fh=self.fh)
        en = self.encode(x)
        de = self.decode(en)
        y = rearrange(de, "b (c fh fw) h w -> b c (h fh) (w fw) ", fw=self.fw, fh=self.fh)
        return y


if __name__ == "__main__":
    seg = SegVitamin(size=(512, 512), fact=(2, 2), in_chans=1, out_channel=13).cuda()

    from ptflops import get_model_complexity_info

    res = get_model_complexity_info(seg, (1, 512, 512), print_per_layer_stat=True, as_strings=True)
    print(res)
