import math
import time

from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights, \
    MOETransformer
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.functional import cosine_similarity


class SFM(nn.Module):
    def __init__(self, dim=512, tokens=6, drop_p=0.3):
        super().__init__()
        self.part_num = tokens
        self.queries = nn.Parameter(torch.randn(tokens, dim) * 0.02)

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(dim * 4, dim),
        )
        self.scale = dim ** -0.5

    def forward(self, x):
        batch = x.shape[0]
        dtype = x.dtype

        attn = torch.matmul(
            self.queries.to(dtype).unsqueeze(0).expand(batch, -1, -1),
            x.transpose(-1, -2)
        )
        attn = (attn * self.scale).softmax(dim=-1)
        out = torch.matmul(attn, x)
        mlp_out = self.mlp(out.float())
        out = out + mlp_out.to(dtype)
        return out

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        self.num = args.cnum

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size)

        # for param in self.base_model.parameters():
        #     param.requires_grad = False  # not update by gradient
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)
        layers = args.moe_layers
        heads = args.moe_heads
        self.STransformerv = MOETransformer(self.embed_dim, layers, heads, attn_mask=None, num_experts=args.num_experts, topk=args.topk, reduction=args.reduction)
        self.STransformert = MOETransformer(self.embed_dim, layers, heads, attn_mask=None, num_experts=args.num_experts, topk=args.topk, reduction=args.reduction)
        for i in range(layers):
            for j in range(args.num_experts):
                nn.init.kaiming_uniform_(self.STransformerv.resblocks[i].feed_forward.experts[j].down.weight,
                                         a=math.sqrt(5))
                nn.init.zeros_(self.STransformerv.resblocks[i].feed_forward.experts[j].down.bias)
                nn.init.zeros_(self.STransformerv.resblocks[i].feed_forward.experts[j].up.weight)
                nn.init.zeros_(self.STransformerv.resblocks[i].feed_forward.experts[j].up.bias)
        for i in range(layers):
            for j in range(args.num_experts):
                nn.init.kaiming_uniform_(self.STransformert.resblocks[i].feed_forward.experts[j].down.weight,
                                         a=math.sqrt(5))
                nn.init.zeros_(self.STransformert.resblocks[i].feed_forward.experts[j].down.bias)
                nn.init.zeros_(self.STransformert.resblocks[i].feed_forward.experts[j].up.weight)
                nn.init.zeros_(self.STransformert.resblocks[i].feed_forward.experts[j].up.bias)

        # self.STransformer = Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)
        # self.STransformer = STransformer(self.embed_dim, layers, heads, attn_mask=None, reduction=args.reduction)
        # scale = self.STransformer.width ** -0.5
        # proj_std = scale * ((2 * self.STransformer.layers) ** -0.5)
        # attn_std = scale
        # fc_std = (2 * self.STransformer.width) ** -0.5
        # for block in self.STransformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        safl_image_feats = self.SFM(x[:, 1:, :])
        safl_image_feats,_ = self.shared_transformer(safl_image_feats,0)
        i_feats = x[:, 0, :].float()
        safl_i_feats = torch.mean(safl_image_feats, dim=1)
        i_feats = safl_i_feats + i_feats
        return i_feats

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        safl_text_feats = self.SFM(x)
        safl_text_feats,_ = self.shared_transformer(safl_text_feats,0)
        t_feats = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        safl_t_feats = torch.mean(safl_text_feats, dim=1)
        t_feats = safl_t_feats + t_feats
        return t_feats

    def encode_image_base(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_text_base(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_s(self, image):
        x = self.base_model.encode_image(image)
        i_feats = x[:, 0, :].float()
        S_image_feats,_ = self.STransformerv(x[:, 1:, :],l_aux=0.0, is_text=False)
        S_image_feats = torch.mean(S_image_feats, dim=1)
        i_feats = S_image_feats + i_feats
        return i_feats

    def encode_text_s(self, text):
        x = self.base_model.encode_text(text)
        t_feats = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        S_text_feats,_ = self.STransformert(x,l_aux=0.0, is_text=True)
        S_text_feats = torch.mean(S_text_feats, dim=1)
        t_feats = S_text_feats + t_feats
        return t_feats

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        g_v  = image_feats[:, 0, :].float()
        g_t = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        # safl_image_feats = self.SFM(image_feats[:, 1:, :])
        # safl_text_feats = self.SFM(text_feats)

        S_image_feats, l_aux_img = self.STransformerv(image_feats[:, 1:, :], l_aux=0.0, is_text=False)
        S_text_feats, l_aux_txt = self.STransformert(text_feats, l_aux=0.0, is_text=True)

        S_i_feats = torch.mean(S_image_feats, dim=1)
        S_t_feats = torch.mean(S_text_feats, dim=1)

        i_feats = S_i_feats + g_v
        t_feats = S_t_feats + g_t

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'aux' in self.current_task:
            l_aux = l_aux_img + l_aux_txt
            # print(f'l_aux:{l_aux}')
            ret.update({'aux_loss': self.args.aux_factor * l_aux})

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(i_feats, t_feats, logit_scale) * 5})

        if 'sdm' in self.current_task:
            ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss': objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})

        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()

            ret.update({'id_loss': objectives.compute_id(image_logits, text_logits, batch['pids'])})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    # model.SFM.mlp = model.SFM.mlp.float()
    return model