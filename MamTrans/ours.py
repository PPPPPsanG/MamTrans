
import os
import math
import torch
import einops
import numpy as np
import torch.nn as nn
from torch import nn, cat
from torch.nn import init
import torch.optim as optim
from einops import rearrange
# from torchsummary import summary
import torch.nn.functional as F
from MamTrans.Transformer import Encoder_Transformer
from MamTrans.Embeddings import PatchEmbeddings, PositionalEmbeddings
from einops.layers.torch import Rearrange, Reduce
from MamTrans.rope_spectral import RotaryEmbedding
from mamba_ssm import Mamba

class Pooling(nn.Module):
    """
    @article{ref-vit,
	title={An image is worth 16x16 words: Transformers for image recognition at scale},
	author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, 
            Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
	journal={arXiv preprint arXiv:2010.11929},
	year={2020}
    }
    """
    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")

        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)


class Classifier(nn.Module):

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MamTrans(nn.Module):
    
    def __init__(self, channels, num_classes, image_size, datasetname, patch_size: int = 1, emb_dim: int = 256, 
                 num_layers: int = 1, num_heads: int = 4, head_dim = 128, hidden_dim: int = 128, pool: str = "mean"):
        super().__init__()

        # Params
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_patch = int(math.sqrt(self.num_patches))
        patch_dim = channels * patch_size ** 2
        self.act = nn.ReLU(inplace=True)

        # Conv Block
        self.conv1 = nn.Conv2d(channels, emb_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(emb_dim)
        self.conv2 = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1, groups=emb_dim, bias=False)
        
        self.bn3 = nn.BatchNorm2d(emb_dim)
        self.conv3 = nn.Conv2d(emb_dim, channels, kernel_size=1, bias=False)
        # Conv Embedded Transformer Encoders----and ----Other Modules
        self.patch_embeddings = PatchEmbeddings(patch_size=patch_size, patch_dim=patch_dim, emb_dim=emb_dim)
        self.pos_embeddings = PositionalEmbeddings(num_pos=self.num_patches, dim=emb_dim)
        self.transformer = Encoder_Transformer(dim=emb_dim, num_layers=num_layers, num_heads=num_heads, 
                                        head_dim=head_dim, hidden_dim=hidden_dim, num_patch=self.num_patch, patch_size=patch_size)
        # self.conv1d = nn.Sequential(nn.Conv1d(self.num_patches+ self.num_patches//4,self.num_patches,kernel_size=1),
        #                             nn.LayerNorm(channels))
        # self.embd = nn.Linear(channels,emb_dim)
        # Graph Conv Layer
        self.patchify = Rearrange("b d c  -> b c d")
        self.bnc = nn.BatchNorm1d(emb_dim)
        self.convc = nn.Conv1d(emb_dim, emb_dim, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(0.8)
        self.num_layers = num_layers
        self.mamba = nn.ModuleList(Mamba(emb_dim) for i in range(num_layers))
        self.mamba_back = nn.ModuleList(Mamba(emb_dim) for i in range(num_layers))
        # Linear Classifier with Pooling, No ClsToken
        self.pool = Pooling(pool=pool)
        self.classifier = Classifier(dim=emb_dim, num_classes=num_classes)
        self.rope = RotaryEmbedding(dim=emb_dim//2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = np.squeeze(x, axis=1)
        
        b, c, hw = x.shape
        x = x.reshape(b,c,int(hw**0.5),int(hw**0.5))
        _,_,h,w, = x.shape
        # Conv Block
        x0 = self.conv1(x)
        x1 = self.conv2(self.act(self.bn2(x0)))
        x = self.conv3(self.act(self.bn3(x1))) + x
        
        # Conv Embedded Transformer Encoders----and ----Other Modules
        x = self.patch_embeddings(x)
        x3 = self.pos_embeddings(x)
        x3 = self.rope.rotate_queries_or_keys(x3)
        x_t = self.transformer(x3)
        
        # BMM
        x_mm =x3
        for i in range(self.num_layers):
            x_m = self.mamba[i](x_mm) 
            x_m_back = self.mamba_back[i](torch.flip(x_mm,[1])) 
            x_mm = x_m + torch.flip(x_m_back,[1]) #+ x_t
        x_c = self.patchify(self.convc(self.act(self.bnc(self.patchify(x_mm))))) + x_mm
        # Linear Classifier with Pooling, No ClsToken
        x = x_t + x_c
        # x_ = x
        # x = x.permute(0,2,1).reshape(b,-1,h,w)
        # x = self.conv_fuse(x)
        # x = x.reshape(b,-1,hw).permute(0,2,1) + x_
        # x = x_t + x0.reshape(b, h * w, self.emb_dim) + x_c
        # x = x_t + x_c + x0.reshape(b, h * w, self.emb_dim)
        x = self.pool(self.dropout(x))
        return self.classifier(x)


if __name__ == '__main__':
    input = torch.randn(size=(100, 200, 11, 11))
    input = input.cuda()
    print("input shape:", input.shape)
    model = MamTrans(channels=200, num_classes=16, image_size=11, datasetname='IndianPines')
    model = model.cuda()
    # summary(model, input.size()[1:])
    print("output shape:", model(input).shape)