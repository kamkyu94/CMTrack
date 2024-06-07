import torch
from fastreid.config import get_cfg
from fastreid.utils.checkpoint import Checkpointer
from fastreid.modeling.meta_arch import build_model


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()
    return cfg


class FastReID(torch.nn.Module):
    def __init__(self, dataset, weights_path):
        super().__init__()
        config_file = "configs/%s/sbs_S50.yml" % dataset
        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])
        self.model = build_model(self.cfg)
        self.model.eval()
        self.model.cuda()

        Checkpointer(self.model).load(weights_path)
        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

    def forward(self, batch):
        # Uses half during training
        batch = batch.half()
        with torch.no_grad():
            return self.model(batch)
