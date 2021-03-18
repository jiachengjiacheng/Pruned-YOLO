import torch

def updateBN(scale, model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.weight.grad.data.add_(scale*torch.sign(m.weight.data))  # L1

def maskBN(model, soft=False, mask_thresh_alpha=0.2):
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            total += m.weight.data.shape[0] # channels numbers
    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size
    bn_mean = torch.mean(bn)
    mask_thresh = mask_thresh_alpha * bn_mean
    print('The number of mask channels in this update is ', torch.sum(bn.abs().clone().le(mask_thresh)))
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.le(mask_thresh).float()

            scale_alpha = -1
            if soft:
                scale_alpha = -0.9
            m.weight.data.add_(scale_alpha*mask*m.weight.data.clone())
            m.bias.data.add_(scale_alpha*mask*m.bias.data.clone())
            m.running_mean.data.add_(scale_alpha*mask*m.running_mean.data.clone())
            m.running_var.data.add_(scale_alpha*mask*m.running_var.data.clone())
