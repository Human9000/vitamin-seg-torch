这是使用vitamin作为backbone的纯分割项目，提供了简单易懂的分割框架（非建造者模式架构，适合个人和小白实验）

```bash
SegVitamin(
  24.65 M, 99.553% Params, 15.64 GMac, 99.771% MACs, 
  (vitamin): VisionTransformer(
    21.79 M, 87.995% Params, 5.79 GMac, 36.917% MACs, 
    (patch_embed): HybridEmbed(
      1.08 M, 4.346% Params, 1.89 GMac, 12.073% MACs, 
      (backbone): MbConvStages(
        1.08 M, 4.346% Params, 1.89 GMac, 12.073% MACs, 
        (stem): Stem(
          39.3 k, 0.159% Params, 643.83 MMac, 4.107% MACs, 
          (conv1): Conv2d(2.37 k, 0.010% Params, 38.8 MMac, 0.247% MACs, 4, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (norm1): LayerNormAct2d(
            0, 0.000% Params, 0.0 Mac, 0.000% MACs, (64,), eps=1e-06, elementwise_affine=True
            (drop): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
            (act): GELU(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          )
          (conv2): Conv2d(36.93 k, 0.149% Params, 605.03 MMac, 3.859% MACs, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (stages): ModuleList(
          (0): Sequential(
            71.3 k, 0.288% Params, 497.55 MMac, 3.174% MACs, 
            (0): MbConvLNBlock(
              35.65 k, 0.144% Params, 351.54 MMac, 2.242% MACs, 
              (shortcut): Downsample2d(
                0, 0.000% Params, 1.05 MMac, 0.007% MACs, 
                (pool): AvgPool2d(0, 0.000% Params, 1.05 MMac, 0.007% MACs, kernel_size=3, stride=2, padding=1)
                (expand): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              )
              (pre_norm): LayerNormAct2d(
                0, 0.000% Params, 0.0 Mac, 0.000% MACs, (64,), eps=1e-06, elementwise_affine=True
                (drop): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
                (act): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              )
              (down): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (conv1_1x1): Conv2d(16.64 k, 0.067% Params, 272.63 MMac, 1.739% MACs, 64, 256, kernel_size=(1, 1), stride=(1, 1))
              (act1): GELU(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (act2): GELU(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (conv2_kxk): Conv2d(2.56 k, 0.010% Params, 10.49 MMac, 0.067% MACs, 256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
              (conv3_1x1): Conv2d(16.45 k, 0.066% Params, 67.37 MMac, 0.430% MACs, 256, 64, kernel_size=(1, 1), stride=(1, 1))
              (drop_path): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
            )
            (1): MbConvLNBlock(
              35.65 k, 0.144% Params, 146.01 MMac, 0.931% MACs, 
              (shortcut): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (pre_norm): LayerNormAct2d(
                0, 0.000% Params, 0.0 Mac, 0.000% MACs, (64,), eps=1e-06, elementwise_affine=True
                (drop): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
                (act): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              )
              (down): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (conv1_1x1): Conv2d(16.64 k, 0.067% Params, 68.16 MMac, 0.435% MACs, 64, 256, kernel_size=(1, 1), stride=(1, 1))
              (act1): GELU(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (act2): GELU(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (conv2_kxk): Conv2d(2.56 k, 0.010% Params, 10.49 MMac, 0.067% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
              (conv3_1x1): Conv2d(16.45 k, 0.066% Params, 67.37 MMac, 0.430% MACs, 256, 64, kernel_size=(1, 1), stride=(1, 1))
              (drop_path): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
            )
          )
          (1): Sequential(
            522.88 k, 2.111% Params, 637.93 MMac, 4.069% MACs, 
            (0): MbConvLNBlock(
              112.38 k, 0.454% Params, 217.58 MMac, 1.388% MACs, 
              (shortcut): Downsample2d(
                8.32 k, 0.034% Params, 8.78 MMac, 0.056% MACs, 
                (pool): AvgPool2d(0, 0.000% Params, 262.14 KMac, 0.002% MACs, kernel_size=3, stride=2, padding=1)
                (expand): Conv2d(8.32 k, 0.034% Params, 8.52 MMac, 0.054% MACs, 64, 128, kernel_size=(1, 1), stride=(1, 1))
              )
              (pre_norm): LayerNormAct2d(
                0, 0.000% Params, 0.0 Mac, 0.000% MACs, (64,), eps=1e-06, elementwise_affine=True
                (drop): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
                (act): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              )
              (down): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (conv1_1x1): Conv2d(33.28 k, 0.134% Params, 136.31 MMac, 0.870% MACs, 64, 512, kernel_size=(1, 1), stride=(1, 1))
              (act1): GELU(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (act2): GELU(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (conv2_kxk): Conv2d(5.12 k, 0.021% Params, 5.24 MMac, 0.033% MACs, 512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
              (conv3_1x1): Conv2d(65.66 k, 0.265% Params, 67.24 MMac, 0.429% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
              (drop_path): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
            )
            (1): MbConvLNBlock(
              136.83 k, 0.553% Params, 140.12 MMac, 0.894% MACs, 
              (shortcut): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (pre_norm): LayerNormAct2d(
                0, 0.000% Params, 0.0 Mac, 0.000% MACs, (128,), eps=1e-06, elementwise_affine=True
                (drop): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
                (act): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              )
              (down): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (conv1_1x1): Conv2d(66.05 k, 0.267% Params, 67.63 MMac, 0.431% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
              (act1): GELU(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (act2): GELU(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (conv2_kxk): Conv2d(5.12 k, 0.021% Params, 5.24 MMac, 0.033% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
              (conv3_1x1): Conv2d(65.66 k, 0.265% Params, 67.24 MMac, 0.429% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
              (drop_path): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
            )
            (2): MbConvLNBlock(
              136.83 k, 0.553% Params, 140.12 MMac, 0.894% MACs, 
              (shortcut): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (pre_norm): LayerNormAct2d(
                0, 0.000% Params, 0.0 Mac, 0.000% MACs, (128,), eps=1e-06, elementwise_affine=True
                (drop): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
                (act): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              )
              (down): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (conv1_1x1): Conv2d(66.05 k, 0.267% Params, 67.63 MMac, 0.431% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
              (act1): GELU(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (act2): GELU(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (conv2_kxk): Conv2d(5.12 k, 0.021% Params, 5.24 MMac, 0.033% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
              (conv3_1x1): Conv2d(65.66 k, 0.265% Params, 67.24 MMac, 0.429% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
              (drop_path): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
            )
            (3): MbConvLNBlock(
              136.83 k, 0.553% Params, 140.12 MMac, 0.894% MACs, 
              (shortcut): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (pre_norm): LayerNormAct2d(
                0, 0.000% Params, 0.0 Mac, 0.000% MACs, (128,), eps=1e-06, elementwise_affine=True
                (drop): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
                (act): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              )
              (down): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (conv1_1x1): Conv2d(66.05 k, 0.267% Params, 67.63 MMac, 0.431% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
              (act1): GELU(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (act2): GELU(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
              (conv2_kxk): Conv2d(5.12 k, 0.021% Params, 5.24 MMac, 0.033% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
              (conv3_1x1): Conv2d(65.66 k, 0.265% Params, 67.24 MMac, 0.429% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
              (drop_path): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
            )
          )
        )
        (pool): StridedConv(
          442.75 k, 1.788% Params, 113.34 MMac, 0.723% MACs, 
          (proj): Conv2d(442.75 k, 1.788% Params, 113.34 MMac, 0.723% MACs, 128, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (norm): LayerNorm2d(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (128,), eps=1e-06, elementwise_affine=True)
        )
      )
      (proj): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    )
    (pos_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
    (patch_drop): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    (norm_pre): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    (blocks): Sequential(
      20.71 M, 83.644% Params, 3.89 GMac, 24.844% MACs, 
      (0): Block(
        1.48 M, 5.975% Params, 278.2 MMac, 1.775% MACs, 
        (norm1): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          591.36 k, 2.388% Params, 50.82 MMac, 0.324% MACs, 
          (qkv): Linear(443.52 k, 1.791% Params, 113.54 MMac, 0.724% MACs, in_features=384, out_features=1152, bias=True)
          (q_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (k_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (attn_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
          (proj): Linear(147.84 k, 0.597% Params, 37.85 MMac, 0.241% MACs, in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
        )
        (ls1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (norm2): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (mlp): GeGluMlp(
          886.66 k, 3.580% Params, 227.18 MMac, 1.449% MACs, 
          (norm): LayerNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
          (act): GELU(0, 0.000% Params, 196.61 KMac, 0.001% MACs, approximate='none')
          (w0): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w1): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w2): Linear(295.3 k, 1.192% Params, 75.6 MMac, 0.482% MACs, in_features=768, out_features=384, bias=True)
        )
        (ls2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (1): Block(
        1.48 M, 5.975% Params, 278.2 MMac, 1.775% MACs, 
        (norm1): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          591.36 k, 2.388% Params, 50.82 MMac, 0.324% MACs, 
          (qkv): Linear(443.52 k, 1.791% Params, 113.54 MMac, 0.724% MACs, in_features=384, out_features=1152, bias=True)
          (q_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (k_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (attn_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
          (proj): Linear(147.84 k, 0.597% Params, 37.85 MMac, 0.241% MACs, in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
        )
        (ls1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (norm2): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (mlp): GeGluMlp(
          886.66 k, 3.580% Params, 227.18 MMac, 1.449% MACs, 
          (norm): LayerNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
          (act): GELU(0, 0.000% Params, 196.61 KMac, 0.001% MACs, approximate='none')
          (w0): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w1): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w2): Linear(295.3 k, 1.192% Params, 75.6 MMac, 0.482% MACs, in_features=768, out_features=384, bias=True)
        )
        (ls2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (2): Block(
        1.48 M, 5.975% Params, 278.2 MMac, 1.775% MACs, 
        (norm1): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          591.36 k, 2.388% Params, 50.82 MMac, 0.324% MACs, 
          (qkv): Linear(443.52 k, 1.791% Params, 113.54 MMac, 0.724% MACs, in_features=384, out_features=1152, bias=True)
          (q_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (k_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (attn_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
          (proj): Linear(147.84 k, 0.597% Params, 37.85 MMac, 0.241% MACs, in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
        )
        (ls1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (norm2): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (mlp): GeGluMlp(
          886.66 k, 3.580% Params, 227.18 MMac, 1.449% MACs, 
          (norm): LayerNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
          (act): GELU(0, 0.000% Params, 196.61 KMac, 0.001% MACs, approximate='none')
          (w0): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w1): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w2): Linear(295.3 k, 1.192% Params, 75.6 MMac, 0.482% MACs, in_features=768, out_features=384, bias=True)
        )
        (ls2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (3): Block(
        1.48 M, 5.975% Params, 278.2 MMac, 1.775% MACs, 
        (norm1): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          591.36 k, 2.388% Params, 50.82 MMac, 0.324% MACs, 
          (qkv): Linear(443.52 k, 1.791% Params, 113.54 MMac, 0.724% MACs, in_features=384, out_features=1152, bias=True)
          (q_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (k_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (attn_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
          (proj): Linear(147.84 k, 0.597% Params, 37.85 MMac, 0.241% MACs, in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
        )
        (ls1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (norm2): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (mlp): GeGluMlp(
          886.66 k, 3.580% Params, 227.18 MMac, 1.449% MACs, 
          (norm): LayerNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
          (act): GELU(0, 0.000% Params, 196.61 KMac, 0.001% MACs, approximate='none')
          (w0): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w1): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w2): Linear(295.3 k, 1.192% Params, 75.6 MMac, 0.482% MACs, in_features=768, out_features=384, bias=True)
        )
        (ls2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (4): Block(
        1.48 M, 5.975% Params, 278.2 MMac, 1.775% MACs, 
        (norm1): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          591.36 k, 2.388% Params, 50.82 MMac, 0.324% MACs, 
          (qkv): Linear(443.52 k, 1.791% Params, 113.54 MMac, 0.724% MACs, in_features=384, out_features=1152, bias=True)
          (q_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (k_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (attn_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
          (proj): Linear(147.84 k, 0.597% Params, 37.85 MMac, 0.241% MACs, in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
        )
        (ls1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (norm2): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (mlp): GeGluMlp(
          886.66 k, 3.580% Params, 227.18 MMac, 1.449% MACs, 
          (norm): LayerNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
          (act): GELU(0, 0.000% Params, 196.61 KMac, 0.001% MACs, approximate='none')
          (w0): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w1): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w2): Linear(295.3 k, 1.192% Params, 75.6 MMac, 0.482% MACs, in_features=768, out_features=384, bias=True)
        )
        (ls2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (5): Block(
        1.48 M, 5.975% Params, 278.2 MMac, 1.775% MACs, 
        (norm1): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          591.36 k, 2.388% Params, 50.82 MMac, 0.324% MACs, 
          (qkv): Linear(443.52 k, 1.791% Params, 113.54 MMac, 0.724% MACs, in_features=384, out_features=1152, bias=True)
          (q_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (k_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (attn_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
          (proj): Linear(147.84 k, 0.597% Params, 37.85 MMac, 0.241% MACs, in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
        )
        (ls1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (norm2): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (mlp): GeGluMlp(
          886.66 k, 3.580% Params, 227.18 MMac, 1.449% MACs, 
          (norm): LayerNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
          (act): GELU(0, 0.000% Params, 196.61 KMac, 0.001% MACs, approximate='none')
          (w0): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w1): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w2): Linear(295.3 k, 1.192% Params, 75.6 MMac, 0.482% MACs, in_features=768, out_features=384, bias=True)
        )
        (ls2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (6): Block(
        1.48 M, 5.975% Params, 278.2 MMac, 1.775% MACs, 
        (norm1): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          591.36 k, 2.388% Params, 50.82 MMac, 0.324% MACs, 
          (qkv): Linear(443.52 k, 1.791% Params, 113.54 MMac, 0.724% MACs, in_features=384, out_features=1152, bias=True)
          (q_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (k_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (attn_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
          (proj): Linear(147.84 k, 0.597% Params, 37.85 MMac, 0.241% MACs, in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
        )
        (ls1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (norm2): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (mlp): GeGluMlp(
          886.66 k, 3.580% Params, 227.18 MMac, 1.449% MACs, 
          (norm): LayerNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
          (act): GELU(0, 0.000% Params, 196.61 KMac, 0.001% MACs, approximate='none')
          (w0): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w1): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w2): Linear(295.3 k, 1.192% Params, 75.6 MMac, 0.482% MACs, in_features=768, out_features=384, bias=True)
        )
        (ls2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (7): Block(
        1.48 M, 5.975% Params, 278.2 MMac, 1.775% MACs, 
        (norm1): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          591.36 k, 2.388% Params, 50.82 MMac, 0.324% MACs, 
          (qkv): Linear(443.52 k, 1.791% Params, 113.54 MMac, 0.724% MACs, in_features=384, out_features=1152, bias=True)
          (q_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (k_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (attn_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
          (proj): Linear(147.84 k, 0.597% Params, 37.85 MMac, 0.241% MACs, in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
        )
        (ls1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (norm2): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (mlp): GeGluMlp(
          886.66 k, 3.580% Params, 227.18 MMac, 1.449% MACs, 
          (norm): LayerNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
          (act): GELU(0, 0.000% Params, 196.61 KMac, 0.001% MACs, approximate='none')
          (w0): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w1): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w2): Linear(295.3 k, 1.192% Params, 75.6 MMac, 0.482% MACs, in_features=768, out_features=384, bias=True)
        )
        (ls2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (8): Block(
        1.48 M, 5.975% Params, 278.2 MMac, 1.775% MACs, 
        (norm1): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          591.36 k, 2.388% Params, 50.82 MMac, 0.324% MACs, 
          (qkv): Linear(443.52 k, 1.791% Params, 113.54 MMac, 0.724% MACs, in_features=384, out_features=1152, bias=True)
          (q_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (k_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (attn_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
          (proj): Linear(147.84 k, 0.597% Params, 37.85 MMac, 0.241% MACs, in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
        )
        (ls1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (norm2): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (mlp): GeGluMlp(
          886.66 k, 3.580% Params, 227.18 MMac, 1.449% MACs, 
          (norm): LayerNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
          (act): GELU(0, 0.000% Params, 196.61 KMac, 0.001% MACs, approximate='none')
          (w0): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w1): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w2): Linear(295.3 k, 1.192% Params, 75.6 MMac, 0.482% MACs, in_features=768, out_features=384, bias=True)
        )
        (ls2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (9): Block(
        1.48 M, 5.975% Params, 278.2 MMac, 1.775% MACs, 
        (norm1): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          591.36 k, 2.388% Params, 50.82 MMac, 0.324% MACs, 
          (qkv): Linear(443.52 k, 1.791% Params, 113.54 MMac, 0.724% MACs, in_features=384, out_features=1152, bias=True)
          (q_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (k_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (attn_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
          (proj): Linear(147.84 k, 0.597% Params, 37.85 MMac, 0.241% MACs, in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
        )
        (ls1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (norm2): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (mlp): GeGluMlp(
          886.66 k, 3.580% Params, 227.18 MMac, 1.449% MACs, 
          (norm): LayerNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
          (act): GELU(0, 0.000% Params, 196.61 KMac, 0.001% MACs, approximate='none')
          (w0): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w1): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w2): Linear(295.3 k, 1.192% Params, 75.6 MMac, 0.482% MACs, in_features=768, out_features=384, bias=True)
        )
        (ls2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (10): Block(
        1.48 M, 5.975% Params, 278.2 MMac, 1.775% MACs, 
        (norm1): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          591.36 k, 2.388% Params, 50.82 MMac, 0.324% MACs, 
          (qkv): Linear(443.52 k, 1.791% Params, 113.54 MMac, 0.724% MACs, in_features=384, out_features=1152, bias=True)
          (q_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (k_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (attn_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
          (proj): Linear(147.84 k, 0.597% Params, 37.85 MMac, 0.241% MACs, in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
        )
        (ls1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (norm2): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (mlp): GeGluMlp(
          886.66 k, 3.580% Params, 227.18 MMac, 1.449% MACs, 
          (norm): LayerNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
          (act): GELU(0, 0.000% Params, 196.61 KMac, 0.001% MACs, approximate='none')
          (w0): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w1): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w2): Linear(295.3 k, 1.192% Params, 75.6 MMac, 0.482% MACs, in_features=768, out_features=384, bias=True)
        )
        (ls2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (11): Block(
        1.48 M, 5.975% Params, 278.2 MMac, 1.775% MACs, 
        (norm1): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          591.36 k, 2.388% Params, 50.82 MMac, 0.324% MACs, 
          (qkv): Linear(443.52 k, 1.791% Params, 113.54 MMac, 0.724% MACs, in_features=384, out_features=1152, bias=True)
          (q_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (k_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (attn_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
          (proj): Linear(147.84 k, 0.597% Params, 37.85 MMac, 0.241% MACs, in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
        )
        (ls1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (norm2): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (mlp): GeGluMlp(
          886.66 k, 3.580% Params, 227.18 MMac, 1.449% MACs, 
          (norm): LayerNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
          (act): GELU(0, 0.000% Params, 196.61 KMac, 0.001% MACs, approximate='none')
          (w0): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w1): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w2): Linear(295.3 k, 1.192% Params, 75.6 MMac, 0.482% MACs, in_features=768, out_features=384, bias=True)
        )
        (ls2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (12): Block(
        1.48 M, 5.975% Params, 278.2 MMac, 1.775% MACs, 
        (norm1): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          591.36 k, 2.388% Params, 50.82 MMac, 0.324% MACs, 
          (qkv): Linear(443.52 k, 1.791% Params, 113.54 MMac, 0.724% MACs, in_features=384, out_features=1152, bias=True)
          (q_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (k_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (attn_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
          (proj): Linear(147.84 k, 0.597% Params, 37.85 MMac, 0.241% MACs, in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
        )
        (ls1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (norm2): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (mlp): GeGluMlp(
          886.66 k, 3.580% Params, 227.18 MMac, 1.449% MACs, 
          (norm): LayerNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
          (act): GELU(0, 0.000% Params, 196.61 KMac, 0.001% MACs, approximate='none')
          (w0): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w1): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w2): Linear(295.3 k, 1.192% Params, 75.6 MMac, 0.482% MACs, in_features=768, out_features=384, bias=True)
        )
        (ls2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (13): Block(
        1.48 M, 5.975% Params, 278.2 MMac, 1.775% MACs, 
        (norm1): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          591.36 k, 2.388% Params, 50.82 MMac, 0.324% MACs, 
          (qkv): Linear(443.52 k, 1.791% Params, 113.54 MMac, 0.724% MACs, in_features=384, out_features=1152, bias=True)
          (q_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (k_norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
          (attn_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
          (proj): Linear(147.84 k, 0.597% Params, 37.85 MMac, 0.241% MACs, in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
        )
        (ls1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path1): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (norm2): LayerNorm(768, 0.003% Params, 98.3 KMac, 0.001% MACs, (384,), eps=1e-06, elementwise_affine=True)
        (mlp): GeGluMlp(
          886.66 k, 3.580% Params, 227.18 MMac, 1.449% MACs, 
          (norm): LayerNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
          (act): GELU(0, 0.000% Params, 196.61 KMac, 0.001% MACs, approximate='none')
          (w0): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w1): Linear(295.68 k, 1.194% Params, 75.69 MMac, 0.483% MACs, in_features=384, out_features=768, bias=True)
          (w2): Linear(295.3 k, 1.192% Params, 75.6 MMac, 0.482% MACs, in_features=768, out_features=384, bias=True)
        )
        (ls2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
        (drop_path2): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
    )
    (norm): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    (fc_norm): LayerNorm(768, 0.003% Params, 384.0 Mac, 0.000% MACs, (384,), eps=1e-06, elementwise_affine=True)
    (head_drop): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.0, inplace=False)
    (head): Linear(385, 0.002% Params, 385.0 Mac, 0.000% MACs, in_features=384, out_features=1, bias=True)
  )
  (up): Sequential(
    2.86 M, 11.558% Params, 9.85 GMac, 62.854% MACs, 
    (0): ConvTranspose2d(1.77 M, 7.150% Params, 1.81 GMac, 11.566% MACs, 2304, 1152, kernel_size=(2, 2), stride=(2, 2), groups=6)
    (1): ChannelShuffle(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    (2): BatchNorm2d(2.3 k, 0.009% Params, 2.36 MMac, 0.015% MACs, 1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU(0, 0.000% Params, 1.18 MMac, 0.008% MACs, )
    (4): ConvTranspose2d(442.94 k, 1.789% Params, 1.81 GMac, 11.573% MACs, 1152, 576, kernel_size=(2, 2), stride=(2, 2), groups=6)
    (5): ChannelShuffle(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    (6): Conv2d(498.24 k, 2.012% Params, 2.04 GMac, 13.018% MACs, 576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6)
    (7): BatchNorm2d(1.15 k, 0.005% Params, 4.72 MMac, 0.030% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(0, 0.000% Params, 2.36 MMac, 0.015% MACs, )
    (9): ConvTranspose2d(110.88 k, 0.448% Params, 1.82 GMac, 11.588% MACs, 576, 288, kernel_size=(2, 2), stride=(2, 2), groups=6)
    (10): ChannelShuffle(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    (11): BatchNorm2d(576, 0.002% Params, 9.44 MMac, 0.060% MACs, 288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): ReLU(0, 0.000% Params, 4.72 MMac, 0.030% MACs, )
    (13): ConvTranspose2d(27.79 k, 0.112% Params, 1.82 GMac, 11.618% MACs, 288, 144, kernel_size=(2, 2), stride=(2, 2), groups=6)
    (14): BatchNorm2d(288, 0.001% Params, 18.87 MMac, 0.120% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (15): ReLU(0, 0.000% Params, 9.44 MMac, 0.060% MACs, )
    (16): Conv2d(7.54 k, 0.030% Params, 494.14 MMac, 3.152% MACs, 144, 52, kernel_size=(1, 1), stride=(1, 1))
  )
)
('15.68 GMac', '24.76 M')
```
