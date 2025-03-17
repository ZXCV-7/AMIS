import torch 
import math
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F
from utils.projector import DimensionalityReducer

class BCEWithLogitsLossWithIgnoreIndex(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, weight=None):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        # targets is B x C x H x W so shape[1] is C

        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        
        if weight is not None:
            loss = loss * weight
        
        if self.reduction == 'mean':
            #return loss.mean()
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            #return loss.sum()
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            #return loss
            return loss * targets.sum(dim=1)

class LabelGuidedOutputDistillation(nn.Module):
      
    def __init__(self, reduction='mean', alpha=1., kd_cil_weights=False):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.kd_cil_weights = kd_cil_weights

    def forward(self, inputs, targets, masks=None):
        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)
        # labels.shape[1] expand tagets.shape[1]
         
        new_labels = torch.zeros_like(outputs)
        new_labels[:, :labels.shape[1]] = labels
        
        unique_labels = torch.unique(masks)
        for i in range(len(unique_labels)):
            if unique_labels[i] != 0 and unique_labels[i] != 255:
                mask = torch.where(masks == unique_labels[i], 1, 0)
                new_labels[:, unique_labels[i]] = mask * labels[:, 0]
                new_labels[:, 0] = (1 - mask) * new_labels[:, 0]
        loss = (outputs * new_labels).mean(dim=1)
        if self.kd_cil_weights:
            w = -(torch.softmax(targets, dim=1) * torch.log_softmax(targets, dim=1)).sum(dim=1) + 1.0
            loss = loss * w[:, None]

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        # outputs.shape is B
        return outputs
    
class AdaptiveMultiScaleDistillation(nn.Module):
    def __init__(self, reduction='mean', alpha=1):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        
    def forward(self, inputs, targets, labels=None, num_classes=None, weights=1, pod=False):
        if pod:
            tot_classes = reduce(lambda a, b: a + b, num_classes)
            new_cls_num = num_classes[-1]
            old_cls_num = tot_classes - new_cls_num
            loss = features_distillation(
                inputs,
                targets,
                labels=labels,
                index_new_class=old_cls_num,
                nb_current_classes=tot_classes,
                nb_new_classes=new_cls_num
            )
        else:
            loss = (inputs - targets) ** 2
        loss = loss * weights * self.alpha
        if self.reduction == 'mean':
            if torch.is_tensor(weights):
                mask = torch.where(weights > 0, 1, 0)
                count = torch.sum(mask.expand_as(loss))
                return torch.sum(loss) / count
            elif weights == 1:
                return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class OtrthogonalLoss(nn.Module):
    def __init__(self, reduction='mean', classes=None):
        super().__init__()
        self.reduction = reduction
        self.classes = classes
    
    def forward(self, class_token, weight=1.):
        class_token = class_token / class_token.norm(dim=-1, keepdim=True)
        class_token_sim = torch.matmul(class_token, class_token.permute(0, 2, 1).detach())
            # class_token_sim.shape is B x C x C
        for i in range(len(class_token_sim)):
            class_token_sim[i].fill_diagonal_(0)
        class_token_sim[:, :self.classes[0]] = 0
    
        non_zero_mask = class_token_sim != 0
        loss_orth = class_token_sim[non_zero_mask].abs().mean()
        
        return loss_orth * weight
    
class KnowledgeDistillationLoss(nn.Module):
      
    def __init__(self, reduction='mean', alpha=1., kd_cil_weights=False):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.kd_cil_weights = kd_cil_weights

    def forward(self, inputs, targets, masks=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])
        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)
        
        loss = (outputs * labels).mean(dim=1)
        if self.kd_cil_weights:
            w = -(torch.softmax(targets, dim=1) * torch.log_softmax(targets, dim=1)).sum(dim=1) + 1.0
            loss = loss * w[:, None]

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs
def features_distillation(
    list_attentions_a,
    list_attentions_b,
    collapse_channels="local",  # 通道池化处理方式
    labels=None,
    index_new_class=None,
    pod_deeplab_mask=False,
    pod_deeplab_mask_factor=None,
    pod_factor=0.01,  # POD损失的权重因子
    prepro="pow",  # 预处理方法
    spp_scales=[1, 2, 4],  # SPP的尺度
    pod_options={"switch": {"after": {"extra_channels": "sum", "factor": 0.0001, "type": "local"}}},
    use_pod_schedule=True,
    nb_current_classes=-1,
    nb_new_classes=-1
):
    device = list_attentions_a[0].device

    assert len(list_attentions_a) == len(list_attentions_b)
#     print("list_attentions_a:", list_attentions_a.shape)
    if pod_deeplab_mask_factor is None:
        pod_deeplab_mask_factor = pod_factor
    normalize = False

    apply_mask = "background"
    upscale_mask_topk = 1
    mask_position = "last"  # 掩码的位置 Others choices "all" "backbone"
    use_adaptative_factor = False
    mix_new_old = None

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        adaptative_pod_factor = 1.0
        difference_function = "frobenius"
        pool = True
        use_adaptative_factor = False
        handle_extra_channels = "sum"
        normalize_per_scale = True

        if pod_options and pod_options.get("switch"):
            if i < len(list_attentions_a) - 1:
                if "before" in pod_options["switch"]:
                    collapse_channels = pod_options["switch"]["before"].get(
                        "type", collapse_channels
                    )
                    pod_factor = pod_options["switch"]["before"].get("factor", pod_factor)
                    normalize = pod_options["switch"]["before"].get("norm", False)
                    prepro = pod_options["switch"]["before"].get("prepro", prepro)
                    use_adaptative_factor = pod_options["switch"]["before"].get(
                        "use_adaptative_factor", use_adaptative_factor
                    )
            else:
                if "after" in pod_options["switch"]:
                    collapse_channels = pod_options["switch"]["after"].get(
                        "type", collapse_channels
                    )
                    pod_factor = pod_options["switch"]["after"].get("factor", pod_factor)
                    normalize = pod_options["switch"]["after"].get("norm", False)
                    prepro = pod_options["switch"]["after"].get("prepro", prepro)

                    apply_mask = pod_options["switch"]["after"].get("apply_mask", apply_mask)
                    upscale_mask_topk = pod_options["switch"]["after"].get(
                        "upscale_mask_topk", upscale_mask_topk
                    )
                    use_adaptative_factor = pod_options["switch"]["after"].get(
                        "use_adaptative_factor", use_adaptative_factor
                    )
                    mix_new_old = pod_options["switch"]["after"].get("mix_new_old", mix_new_old)

                    handle_extra_channels = pod_options["switch"]["after"].get(
                        "extra_channels", handle_extra_channels
                    )
                    spp_scales = pod_options["switch"]["after"].get(
                        "spp_scales", spp_scales
                    )
                    use_pod_schedule = pod_options["switch"]["after"].get(
                        "use_pod_schedule", use_pod_schedule
                    )

            mask_position = pod_options["switch"].get("mask_position", mask_position)
            normalize_per_scale = pod_options["switch"].get(
                "normalize_per_scale", normalize_per_scale
            )
            pool = pod_options.get("pool", pool)

        if a.shape[1] != b.shape[1]:
            assert i == len(list_attentions_a) - 1
            assert a.shape[0] == b.shape[0]
            assert a.shape[2] == b.shape[2]
            assert a.shape[3] == b.shape[3]

            assert handle_extra_channels in ("trim", "sum"), handle_extra_channels

            if handle_extra_channels == "sum":
                _b = torch.zeros_like(a).to(a.dtype).to(a.device)
                _b[:, 0] = b[:, 0] + b[:, index_new_class + 1:].sum(dim=1)
                _b[:, 1:] = b[:, 1:index_new_class + 1]
                b = _b
            elif handle_extra_channels == "trim":
                b = b[:, :index_new_class + 1]
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if not pod_deeplab_mask and use_adaptative_factor:
            adaptative_pod_factor = (labels == 0).float().mean()

        if prepro == "pow":
            a = torch.pow(a, 2)
            b = torch.pow(b, 2)
        elif prepro == "none":
            pass
        elif prepro == "abs":
            a = torch.abs(a, 2)
            b = torch.abs(b, 2)
        elif prepro == "relu":
            a = torch.clamp(a, min=0.)
            b = torch.clamp(b, min=0.)

        if collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)

        elif collapse_channels == "local":
                a = _local_pod(
                    a, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
                b = _local_pod(
                    b, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if i == len(list_attentions_a) - 1 and pod_options is not None:
            if "difference_function" in pod_options:
                difference_function = pod_options["difference_function"]
        elif pod_options is not None:
            if "difference_function_all" in pod_options:
                difference_function = pod_options["difference_function_all"]

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        if difference_function == "frobenius":
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.frobenius_norm(aa - bb, dim=-1) for aa, bb in zip(a, b)]
                ).to(device)
            else:
                layer_loss = torch.frobenius_norm(a - b, dim=-1)
        elif difference_function == "frobenius_mix":
            layer_loss_old = torch.frobenius_norm(a[0] - b[0], dim=-1)
            layer_loss_new = torch.frobenius_norm(a[1] - b[1], dim=-1)

            layer_loss = mix_new_old * layer_loss_old + (1 - mix_new_old) * layer_loss_new
        elif difference_function == "l1":
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.norm(aa - bb, p=1, dim=-1) for aa, bb in zip(a, b)]
                ).to(device)
            else:
                layer_loss = torch.norm(a - b, p=1, dim=-1)
        elif difference_function == "kl":
            d1, d2, d3 = a.shape
            a = (a.view(d1 * d2, d3) + 1e-8).log()
            b = b.view(d1 * d2, d3) + 1e-8

            layer_loss = F.kl_div(a, b, reduction="none").view(d1, d2, d3).sum(dim=(1, 2))
        elif difference_function == "bce":
            d1, d2, d3 = a.shape
            layer_loss = bce(a.view(d1 * d2, d3), b.view(d1 * d2, d3)).view(d1, d2,
                                                                            d3).mean(dim=(1, 2))
        else:
            raise NotImplementedError(f"Unknown difference_function={difference_function}")

        assert torch.isfinite(layer_loss).all(), layer_loss
        assert (layer_loss >= 0.).all(), layer_loss

        layer_loss = torch.mean(adaptative_pod_factor * layer_loss)
        if pod_factor <= 0.:
            continue

        layer_loss = pod_factor * layer_loss
        if use_pod_schedule:
            layer_loss = layer_loss * math.sqrt(nb_current_classes / nb_new_classes)
        loss += layer_loss

    return loss / len(list_attentions_a)

def bce(x, y):
    return -(y * torch.log(x + 1e-6) + (1 - y) * torch.log((1 - x) + 1e-6))

# x = b * c * h * w
def _local_pod(x, spp_scales=[1, 2, 4], normalize=False, normalize_per_scale=False):
    b = x.shape[0]
    w = x.shape[-1]
    emb = []

    for scale_index, scale in enumerate(spp_scales):
        k = w // scale

        nb_regions = scale**2

        for i in range(scale):
            for j in range(scale):

                tensor = x[..., i * k:(i + 1) * k, j * k:(j + 1) * k]
                horizontal_pool = tensor.mean(dim=1).view(b, -1)
                vertical_pool = tensor.mean(dim=1).view(b, -1)

                if normalize_per_scale is True:
                    horizontal_pool = horizontal_pool / nb_regions
                    vertical_pool = vertical_pool / nb_regions
                elif normalize_per_scale == "spm":
                    if scale_index == 0:
                        factor = 2 ** (len(spp_scales) - 1)
                    else:
                        factor = 2 ** (len(spp_scales) - scale_index)
                    horizontal_pool = horizontal_pool / factor
                    vertical_pool = vertical_pool / factor

                if normalize:
                    horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
                    vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

                emb.append(horizontal_pool)
                emb.append(vertical_pool)

    return torch.cat(emb, dim=1)

def class_embedding_FOD(e_t_c, e_t_minus_1_c, old_cls_num):
    beta = torch.ones(old_cls_num, dtype=torch.float32)
    device = e_t_c.device
    e_t_minus_1_c = e_t_minus_1_c.to(device)
    beta = beta.to(device)

    target_dim = e_t_minus_1_c.size(1)
    e_t_c = e_t_c[:, :target_dim].to(device)
    cos_sim = F.cosine_similarity(e_t_c, e_t_minus_1_c, dim=-1)

    distillation_loss = (1.0 / old_cls_num) * torch.sum(beta * (1 - cos_sim))
    return distillation_loss