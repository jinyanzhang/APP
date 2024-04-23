from utils.activation import SymSum
from torch import nn
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_model(args):
    act_mapper = {
        "gelu": nn.GELU,
        'relu': nn.ReLU,
        'symsum': SymSum
    }

    if args.model_name == "MotionAGFormer":
        from model.MotionAGFormer import MotionAGFormer
        model = MotionAGFormer(n_layers=args.n_layers,
                               dim_in=args.dim_in,
                               dim_feat=args.dim_feat,
                               dim_rep=args.dim_rep,
                               dim_out=args.dim_out,
                               mlp_ratio=args.mlp_ratio,
                               act_layer=act_mapper[args.act_layer],
                               attn_drop=args.attn_drop,
                               drop=args.drop,
                               drop_path=args.drop_path,
                               use_layer_scale=args.use_layer_scale,
                               layer_scale_init_value=args.layer_scale_init_value,
                               use_adaptive_fusion=args.use_adaptive_fusion,
                               num_heads=args.num_heads,
                               qkv_bias=args.qkv_bias,
                               qkv_scale=args.qkv_scale,
                               hierarchical=args.hierarchical,
                               num_joints=args.num_joints,
                               use_temporal_similarity=args.use_temporal_similarity,
                               temporal_connection_len=args.temporal_connection_len,
                               use_tcn=args.use_tcn,
                               graph_only=args.graph_only,
                               neighbour_num=args.neighbour_num,
                               n_frames=args.n_frames)
    elif args.model_name == 'MotionBERT':
        from model.MotionBERT import MotionBERT
        assert args.n_frames == args.maxlen
        model = MotionBERT(
            dim_feat=args.dim_feat,
            dim_rep=args.dim_rep,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            num_joints=args.num_joints,
            maxlen=args.maxlen,
            att_fuse=args.att_fuse
        )
    elif args.model_name == 'Transformer':
        from model.Transformer import Transformer
        model = Transformer(
            dim_in=args.dim_in,
            dim_feat=args.dim_feat,
            dim_rep=args.dim_rep,
            dim_out=args.dim_out,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            qkv_bias=args.qkv_bias,
            qk_scale=args.qk_scale,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            drop_path_rate=args.drop_path_rate,
            att_fuse=args.att_fuse,
            mode=args.mode,
            
            num_joints=args.num_joints,
            n_frames=args.n_frames
        )
    elif args.model_name == 'HoT':
        from model.HoT import HoT
        model = HoT(
            maxlen=args.maxlen,
            dim_feat=args.dim_feat,
            mlp_ratio=args.mlp_ratio,
            depth=args.depth,
            dim_rep=args.dim_rep,
            num_heads=args.num_heads
        )
    elif args.model_name == 'MixSTE':
        from model.MixSTE import MixSTE
        model = MixSTE(
            in_chans=args.in_chans,
            embed_dim_ratio=args.embed_dim_ratio,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            qkv_bias=args.qkv_bias,
            qk_scale=args.qk_scale,
            drop_path_rate=args.drop_path_rate,
            
            num_joints=args.num_joints,
            n_frames=args.n_frames
        )
    elif args.model_name == 'STCFormer':
        from model.STCFormer import STCFormer
        model = STCFormer(
            dim_in=2,
            layers=6,
            d_hid=512,
            
            num_joints=args.num_joints,
            n_frames=args.n_frames
        )
    elif args.model_name == 'AdaptivePosePoolingv1':
        from model.AdaptivePosePoolingv1 import AdaptivePosePoolingv1
        args.encoder.enc_act_layer = act_mapper[args.encoder.enc_act_layer]
        args.decoder.dec_act_layer = act_mapper[args.decoder.dec_act_layer]
        model = AdaptivePosePoolingv1(**args.encoder, **args.decoder, num_joints=args.num_joints, n_frames=args.n_frames)
    elif args.model_name == 'AdaptivePosePoolingv2':
        from model.AdaptivePosePoolingv2 import AdaptivePosePoolingv2
        args.model.act_layer = act_mapper[args.model.act_layer]
        model = AdaptivePosePoolingv2(**args.model, num_joints=args.num_joints, n_frames=args.n_frames)
    elif args.model_name == 'AdaptivePosePoolingv2_image':
        from model.AdaptivePosePoolingv2_image import AdaptivePosePoolingv2
        args.model.act_layer = act_mapper[args.model.act_layer]
        model = AdaptivePosePoolingv2(**args.model, num_joints=args.num_joints, n_frames=args.n_frames)
    elif args.model_name == 'AdaptivePosePoolingv3':
        from model.AdaptivePosePoolingv3 import AdaptivePosePoolingv3
        args.model.act_layer = act_mapper[args.model.act_layer]
        model = AdaptivePosePoolingv3(**args.model, num_joints=args.num_joints, n_frames=args.n_frames)
    else:
        raise Exception("Undefined model name")

    return model


def load_pretrained_weights(model, checkpoint):
    """
    Load pretrained weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - checkpoint (dict): the checkpoint
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict:
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
    print(f'[INFO] (load_pretrained_weights) {len(matched_layers)} layers are loaded')
    print(f'[INFO] (load_pretrained_weights) {len(discarded_layers)} layers are discared')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def decay_lr_exponentially(lr, lr_decay, optimizer):
    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return lr