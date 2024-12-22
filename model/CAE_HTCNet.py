import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, dim_embedding, max_len):
        super(PositionalEncoding, self).__init__()

        self.pe = torch.zeros(max_len, dim_embedding) #60 64
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 60 1
        div_term = torch.exp(torch.arange(0, dim_embedding, 2).float() * (-math.log(10000.0) / dim_embedding))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):  # b c d 32 5 64
        x = x + self.pe[:x.size(-2), :].to(device)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        if isinstance(out, tuple):
            out_tensor, *rest = out
            out = (out_tensor + x, *rest)
        else:
            out = out + x
        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.scale_ = self.inner_dim ** -0.5
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        attn_ = torch.einsum('bij,bjk->bik', qkv[0], qkv[1].permute(0,2,1)) * self.scale_
        attn_ = attn_.softmax(dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, attn_

class Transformer(nn.Module):
    def __init__(self, dim, encoder, heads, dim_head, mlp_head, dropout):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(encoder):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))
        self.attention_weights = []

    def forward(self, x, mask=None):

        self.attention_weights = []
        for attn, ff in self.layers:
            x_out = attn(x, mask=mask)
            if isinstance(x_out, tuple):
                x, attn_weights = x_out
                self.attention_weights.append(attn_weights.clone())
            else:
                x = x_out
            x = ff(x)

        return x

class GSE(nn.Module):
    def __init__(self, channels,dim, local_band_num=3):
        super(GSE, self).__init__()

        self.channels = channels
        self.L = local_band_num
        self.dim = dim
        self.W = nn.Parameter(torch.randn(channels, self.L, dim))
        self.emb = nn.Linear(self.L, dim, bias=False)
        self.pos_embedding = PositionalEncoding(dim, channels + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # Padding
        if local_band_num % 2 == 0:
            self.pad = local_band_num // 2
        elif local_band_num % 2 == 1:
            self.pad = (local_band_num - 1) // 2

    def forward(self, x):

        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = x.unfold(1, self.L, 1)
        b, n, l = x.shape
        x_re = x.contiguous().view(b * n, l)
        w_reshaped = self.W.unsqueeze(0).expand(b, -1, -1, -1).contiguous().view(-1, self.L, self.dim)
        output = torch.bmm(x_re.unsqueeze(1), w_reshaped).squeeze(1)
        x = output.view(b, n, self.dim)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding(x)

        return x

class ViT(nn.Module):
    def __init__(self, in_channel, dim, encoder, heads, mlp_dim, dim_head, dropout, emb_dropout):
        super().__init__()
        self.dim = dim
        self.gse = GSE(in_channel,dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, encoder, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

    def forward(self, x, mask=None):
        x = self.gse(x)
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = self.to_latent(x[:, 0])
        attention_weights = [attn[:, 1:, 1:] for attn in self.transformer.attention_weights]

        return x, attention_weights

class EduCosSim(nn.Module):
    def __init__(self):
        super(EduCosSim, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, patch, method='cos'):

        batch, c, h, w = patch.size()
        patch_re = patch.view(batch, c, h * w).permute(0, 2, 1)
        center_idx = (h * w - 1) // 2
        center_spec_vector = patch_re[:, center_idx, :].unsqueeze(1)
        if method == 'edu':
            patch_center_expanded = center_spec_vector.expand(batch, h * w, c)
            e_dist = torch.norm(patch_center_expanded - patch_re, dim=2, p=2)
            sim_final = 1 / (1 + e_dist)
        elif method == 'cos':
            sim_final = torch.cosine_similarity(center_spec_vector, patch_re, dim=2)
        elif method == 'scm':
            center_mean = center_spec_vector.mean(dim=2, keepdim=True)
            patch_mean = patch_re.mean(dim=2, keepdim=True)
            x_diff = center_spec_vector - center_mean
            y_diff = patch_re - patch_mean
            numerator = torch.sum(x_diff * y_diff, dim=2)
            denominator = torch.sqrt(torch.sum(x_diff ** 2, dim=2) * torch.sum(y_diff ** 2, dim=2))
            sim_final = numerator / denominator
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'edu', 'cos', or 'scm'.")

        return sim_final.view(batch, h, w)

class SRAE(nn.Module):
    def __init__(self):
        super(SRAE, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float), requires_grad=True)
        self.edu_cos_sim = EduCosSim()

    def forward(self, patch):

        edu_sim = self.edu_cos_sim(patch, method='edu')
        edu_sim = edu_sim.softmax(dim=-1)
        scm_sim = self.edu_cos_sim(patch, method='scm')
        scm_sim = scm_sim.softmax(dim=-1)
        alpha = torch.sigmoid(self.alpha)
        weighted_sim = alpha * edu_sim + (1 - alpha) * scm_sim
        patch_new = patch * weighted_sim.unsqueeze(1)

        return patch_new
class CSS_Conv(nn.Module):
    def __init__(self, channel):
        super(CSS_Conv, self).__init__()

        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.GELU()
        )
        self.pointwise_conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.GELU()
        )
        self.depthwise_conv_res = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel,bias=False),
            nn.BatchNorm2d(channel),
            nn.GELU()
        )

    def forward(self, patch):

        patch_new1 = self.pointwise_conv(patch)
        patch_new1 = self.pointwise_conv2(patch_new1)
        patch_new_res = self.depthwise_conv_res(patch_new1 )+ patch_new1

        return patch_new_res

class CAE(nn.Module):
    def __init__(self, in_channel, num_encoder):
        super(CAE, self).__init__()

        self.w = nn.Parameter(torch.zeros(num_encoder))
        self.softmax = nn.Softmax(dim=-2)
        self.Conv = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=1, bias=False)

    def forward(self, patch, attention_weights):

        w = self.w.softmax(dim=-1)
        SAM = sum([attn*w[idx]  for idx, attn in enumerate(attention_weights)])
        SAW = self.softmax(SAM).unsqueeze(-1)
        SAW = self.Conv(SAW).permute(0,2,1,3)
        out = patch * SAW

        return out

class CAE_HTCNet(nn.Module):
    def __init__(self, in_channel, num_classes, dim_emb=64, num_encoder=1, heads=4, mlp_dim=128,
                 dim_head=16, dropout=0.2, emb_dropout=0.3):
        super(CAE_HTCNet, self).__init__()

        self.beta = nn.Parameter(torch.tensor(0.5, dtype=torch.float), requires_grad=True)
        self.SRAE = SRAE()
        self.CSS_Conv = CSS_Conv(in_channel)
        self.spectral_branch = ViT(in_channel, dim_emb, num_encoder, heads, mlp_dim,
                              dim_head, dropout, emb_dropout)
        self.CAE = CAE(in_channel, num_encoder)
        self.fc1 = nn.Linear(dim_emb + in_channel, (dim_emb + in_channel) // 2)
        self.dropout = nn.Dropout(emb_dropout)
        self.classifier = nn.Linear((dim_emb + in_channel) // 2, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

    def forward(self, patch):

        patch = patch.permute(0,3,1,2)
        b,c,h,w = patch.shape
        vector = patch[:,:,h//2,w//2]
        # Spectral brach
        vector_spe, SAM = self.spectral_branch(vector)
        # Spatial branch
        patch_srae = self.SRAE(patch)
        beta = torch.sigmoid(self.beta)
            # CAE
        patch_cae = self.CAE((beta * patch_srae + (1-beta) * patch), SAM)
        patch_fin = self.CSS_Conv(patch_srae + patch_cae) + patch
        vector_spa = self.flatten(self.avgpool(patch_fin))
        out_fuse = torch.cat((vector_spe, vector_spa), dim=-1)
        out = self.dropout(torch.sigmoid(self.fc1(out_fuse)))
        out = self.classifier(out)

        return out