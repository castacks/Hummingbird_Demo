import torch
import torch.nn as nn
from collections import OrderedDict

from ..encoders import twins_svt_large
from .encoder import MemoryEncoder
from .decoder import MemoryDecoder
from .cnn import BasicEncoder
from ....core.utils.utils import InputPadder


class FlowFormer(nn.Module):
    def __init__(self, cfg, device: str, use_inference_jit=False):
        super(FlowFormer, self).__init__()
        self.device = device
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg, device, use_inference_jit)
        self.memory_decoder = MemoryDecoder(cfg)
        if use_inference_jit:
            self.freeze_handle  = self.memory_decoder.register_load_state_dict_post_hook(self.__freeze_decoder)

        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')

    @staticmethod
    def __freeze_decoder(module, _):
        if not module.use_jit_inference: return
        module.memory_decoder = torch.jit.optimize_for_inference(
            torch.jit.script(module.memory_decoder)
        )
        module.freeze_handle.remove()   # Should not be triggered twice.

    def forward(self, image1, image2):
        # Following https://github.com/princeton-vl/RAFT/
        image1 = (2 * image1) - 1.0
        image2 = (2 * image2) - 1.0

        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)

        cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions = self.memory_decoder(cost_memory, context, data["cost_maps"], self.cfg.query_latent_dim, flow_init=None)

        return flow_predictions

    @torch.no_grad()
    @torch.inference_mode()
    def inference(self, image1, image2):
        image1, image2 = image1.to(self.device), image2.to(self.device)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = self(image1, image2)

        flow_pre = padder.unpad(flow_pre[0])
        flow = flow_pre[0]
        return flow, torch.empty(0)

    def load_ddp_state_dict(self, ckpt: OrderedDict):
        cvt_ckpt = OrderedDict()
        for k in ckpt:
            if k.startswith("module."):
                cvt_ckpt[k[7:]] = ckpt[k]
            else:
                cvt_ckpt[k] = ckpt[k]
        self.load_state_dict(cvt_ckpt)

