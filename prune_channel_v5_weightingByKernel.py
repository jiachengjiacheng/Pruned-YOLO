import argparse
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn as nn

import test  # import test.py to get mAP after each epoch
from models.yolo import *
from models.experimental import *
from models.common import *
from utils.datasets import *
from utils.general import *
from utils.torch_utils import *

def channel_count_rough(model):
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            total += m.weight.data.shape[0] # channels numbers
    return total


def grab_thresh(model, overall_ratio, save, weights):
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            total += m.weight.data.shape[0] # channels numbers
    bn = torch.zeros(total)
    index = 0
    last_m_weight = None
    bn_layer_mean_list, bn_layer_var_list = [], []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_m_weight = m.weight.data.abs().clone()
        if isinstance(m, torch.nn.BatchNorm2d):
            kernel_weight = last_m_weight
            weight_alpha = torch.mean( kernel_weight.view(kernel_weight.size()[0], -1), dim=1 )
            bn_weight = m.weight.data.abs().clone()
            assert weight_alpha.size() == bn_weight.size()
            weight_copy = 10 * weight_alpha * bn_weight

            size = m.weight.data.shape[0]
            bn[index:(index+size)] = weight_copy
            bn_layer_mean_list.append(torch.mean(bn[index:(index+size)]))
            bn_layer_var_list.append(torch.var(bn[index:(index+size)]))
            index += size
    sorted_bn, sorted_index = torch.sort(bn)
    thresh_index = int(total*overall_ratio)
    thresh = sorted_bn[thresh_index].to(device)
    print('prune ratio is {}, prune thresh of BN is {}'.format(overall_ratio, thresh))
    bn_layer_mean = torch.Tensor(bn_layer_mean_list).numpy().tolist()
    bn_layer_var = [i*10 for i in torch.Tensor(bn_layer_var_list).numpy().tolist()]
    plot_pic = True
    if plot_pic:
        import matplotlib.pyplot as plt
        x = [ i for i in range(len(bn_layer_mean)) ]
        plt.figure(figsize=(10,8))
        plt.ylim(0, 2)
        plt.plot(x, bn_layer_mean, 'r', linewidth=1, label='mean')
        plt.plot(x, bn_layer_var, 'g', linewidth=1, label='var')
        plt.grid(True, linestyle='-.')
        plt.xlabel("Layer index")
        plt.ylabel("BN mean / var")
        plt.legend(ncol=1)
        epoch = os.path.splitext(weights)[0].split('_')[-1]
        plt.savefig(os.path.join(save, "layer_bn_epoch_%s.jpg" % epoch))
    print("--"*30)
    return thresh

def parse_model(d):
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    assert gd == 1. and gw == 1.
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    len_backbone = len(d['backbone'])

    backbone_ifo, head_ifo = {}, {}
    for i, (f, n, m, args) in enumerate(d['backbone']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        backbone_ifo[i] = [f, n, m, args]
    for i, (f, n, m, args) in enumerate(d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        head_ifo[i] = [f, n, m, args]
    #print(backbone_ifo, '\n', head_ifo)
    return backbone_ifo, head_ifo


def cal_mask_weightingByKernel(thresh, perlayer_ratio, block_conv, block_bn):
    assert isinstance(block_conv, torch.nn.Conv2d) and isinstance(block_bn, torch.nn.BatchNorm2d)
    kernel_weight = block_conv.weight.data.abs().clone()
    weight_alpha = torch.mean( kernel_weight.view(kernel_weight.size()[0], -1), dim=1 )
    bn_weight = block_bn.weight.data.abs().clone()
    assert weight_alpha.size() == bn_weight.size()
    weight_copy = 10 * weight_alpha * bn_weight

    channels = weight_copy.shape[0]
    min_channel_num = int(channels * perlayer_ratio) if int(channels * perlayer_ratio) > 0 else 1
    if min_channel_num < 2:
        min_channel_num = 2
    mask = weight_copy.gt(thresh).float().to(device)
    if int(torch.sum(mask)) < min_channel_num:
         _, sorted_index_weights = torch.sort(weight_copy,descending=True)
         mask[sorted_index_weights[:min_channel_num]]=1.
    return mask

def cal_mask_weight(weight_copy, thresh, perlayer_ratio):
    channels = weight_copy.shape[0]
    min_channel_num = int(channels * perlayer_ratio) if int(channels * perlayer_ratio) > 0 else 1
    if min_channel_num < 2:
        min_channel_num = 2
    mask = weight_copy.gt(thresh).float().to(device)
    if int(torch.sum(mask)) < min_channel_num:
         _, sorted_index_weights = torch.sort(weight_copy,descending=True)
         mask[sorted_index_weights[:min_channel_num]]=1.
    return mask


def extract_weights(template, weights, destination, mask_list):
    with open(template) as f:
        model_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    backbone_len = len(model_dict['backbone'])
    model_ifo = model_dict['backbone'] + model_dict['head']
    #print(model_ifo)
    save_path = destination.replace('.yaml', '.pt')
    print(save_path)
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    n_state_dict = {}
    # load model
    model_items = None
    if isinstance(ckpt['model'], Model):
        model_items = ckpt['model'].float().state_dict().items()
    else:
        model_items = ckpt['model'].items()
    for k, v in model_items:
        exclude = ['anchor']
        if any(x in k for x in exclude):
           n_state_dict[k] = v
           continue
        layer_idx = int( k.split('.')[1] )
        #layer_mask = mask_list[layer_idx]
        pre_layer = model_ifo[layer_idx][0]  # remove int
        layer_number = int(model_ifo[layer_idx][1])
        layer_type = model_ifo[layer_idx][2]
        layer_args = model_ifo[layer_idx][3]
        #print(layer_type)
        mask_in, mask_out = None, None
        tweight = None
        if layer_type in ['Focus']:
            mask_in, mask_out = torch.Tensor([1.]*12).to(device), mask_list[layer_idx][0]
            idx1 = np.squeeze(np.argwhere(np.asarray(mask_in.cpu().numpy())))
            idx2 = np.squeeze(np.argwhere(np.asarray(mask_out.cpu().numpy())))
            if 'conv.weight' in k:
                tweight = v[idx2.tolist()][:, idx1.tolist(), :, :].clone()
            elif 'bn.weight' in k or 'bn.bias' in k or 'bn.running_mean' in k or 'bn.running_var' in k:
                tweight = v[idx2.tolist()].clone()
            else:
                tweight = v
        elif layer_type in ['Conv']:
            mask_in = mask_list[layer_idx+pre_layer][-1]
            mask_out = mask_list[layer_idx][0]
            idx1 = np.squeeze(np.argwhere(np.asarray(mask_in.cpu().numpy())))
            idx2 = np.squeeze(np.argwhere(np.asarray(mask_out.cpu().numpy())))
            if 'conv.weight' in k:
                tweight = v[idx2.tolist()][:, idx1.tolist(), :, :].clone()
            elif 'bn.weight' in k or 'bn.bias' in k or 'bn.running_mean' in k or 'bn.running_var' in k:
                tweight = v[idx2.tolist()].clone()
            else:
                tweight = v
        elif layer_type in ['C3']:
            mask_in = mask_list[layer_idx-1][-1]
            mask_out = mask_list[layer_idx]
            assert len(mask_out) == 4

            idx_in = np.squeeze(np.argwhere(np.asarray(mask_in.cpu().numpy())))
            idx1_list = [np.squeeze(np.argwhere(np.asarray(mask_out_t.cpu().numpy()))) for mask_out_t in mask_out[0]]
            idx2_list = [np.squeeze(np.argwhere(np.asarray(mask_out_t.cpu().numpy()))) for mask_out_t in mask_out[1]]
            idx_cv1_out = idx2_list[0]
            idx_cv2_out =  np.squeeze(np.argwhere(np.asarray(mask_out[2].cpu().numpy())))
            idx_cv3_in  =  np.squeeze(np.argwhere(np.asarray( torch.cat([mask_out[1][-1], mask_out[2]], dim=0).cpu().numpy())))
            idx_cv3_out =  np.squeeze(np.argwhere(np.asarray(mask_out[3].cpu().numpy())))
            assert len(idx1_list)+1 == len(idx2_list)

            if '.m.' in k:
                cv_idx = int(k.split('.cv')[0][-1])
                idx1 = idx2_list[cv_idx]
                idx2 = idx1_list[cv_idx]
                idx3 = idx2_list[cv_idx+1]
                if 'cv1.conv.weight' in k:
                    tweight = v[idx2.tolist()][:, idx1.tolist(), :, :].clone()
                elif 'cv1.bn.weight' in k or 'cv1.bn.bias' in k or 'cv1.bn.running_mean' in k or 'cv1.bn.running_var' in k:
                    tweight = v[idx2.tolist()].clone()
                elif 'cv2.conv.weight' in k:
                    tweight = v[idx3.tolist()][:, idx2.tolist(), :, :].clone()
                elif 'cv2.bn.weight' in k or 'cv2.bn.bias' in k or 'cv2.bn.running_mean' in k or 'cv2.bn.running_var' in k:
                    tweight = v[idx3.tolist()].clone()
                elif 'cv1.bn.num_batches_tracked' in k or 'cv2.bn.num_batches_tracked' in k:
                    tweight = v
                else:
                    print(k)
                    assert k == 'not support'
            else:
                if 'cv1.conv.weight' in k:
                    tweight = v[idx_cv1_out.tolist()][:, idx_in.tolist(), :, :].clone()
                elif 'cv1.bn.weight' in k or 'cv1.bn.bias' in k or 'cv1.bn.running_mean' in k or 'cv1.bn.running_var' in k:
                    tweight = v[idx_cv1_out.tolist()].clone()
                elif 'cv2.conv.weight' in k:
                    tweight = v[idx_cv2_out.tolist()][:, idx_in.tolist(), :, :].clone()
                elif 'cv2.bn.weight' in k or 'cv2.bn.bias' in k or 'cv2.bn.running_mean' in k or 'cv2.bn.running_var' in k:
                    tweight = v[idx_cv2_out.tolist()].clone()
                elif 'cv3.conv.weight' in k:
                    tweight = v[idx_cv3_out.tolist()][:, idx_cv3_in.tolist(), :, :].clone()
                elif 'cv3.bn.weight' in k or 'cv3.bn.bias' in k or 'cv3.bn.running_mean' in k or 'cv3.bn.running_var' in k:
                    tweight = v[idx_cv3_out.tolist()].clone()
                elif 'cv1.bn.num_batches_tracked' in k or 'cv2.bn.num_batches_tracked' in k or 'cv3.bn.num_batches_tracked' in k:
                    tweight = v
                else:
                    print(k)
                    assert k == 'not support'

        elif layer_type in ['SPP']:
            mask_in_cv1 = mask_list[layer_idx+pre_layer][-1]
            mask_out_cv1 = mask_list[layer_idx][0]
            mask_in_cv2 = torch.cat([mask_out_cv1, mask_out_cv1, mask_out_cv1, mask_out_cv1], dim=0)
            mask_out_cv2 = mask_list[layer_idx][1]

            idx1 = np.squeeze(np.argwhere(np.asarray(mask_in_cv1.cpu().numpy())))
            idx2 = np.squeeze(np.argwhere(np.asarray(mask_out_cv1.cpu().numpy())))
            idx3 = np.squeeze(np.argwhere(np.asarray(mask_in_cv2.cpu().numpy())))
            idx4 = np.squeeze(np.argwhere(np.asarray(mask_out_cv2.cpu().numpy())))

            if 'cv1.conv.weight' in k:
                tweight = v[idx2.tolist()][:, idx1.tolist(), :, :].clone()
            elif 'cv1.bn.weight' in k or 'cv1.bn.bias' in k or 'cv1.bn.running_mean' in k or 'cv1.bn.running_var' in k:
                tweight = v[idx2.tolist()].clone()
            elif 'cv2.conv.weight' in k:
                tweight = v[idx4.tolist()][:, idx3.tolist(), :, :].clone()
            elif 'cv2.bn.weight' in k or 'cv2.bn.bias' in k or 'cv2.bn.running_mean' in k or 'cv2.bn.running_var' in k:
                tweight = v[idx4.tolist()].clone()
            elif 'cv1.bn.num_batches_tracked' in k or 'cv2.bn.num_batches_tracked' in k:
                tweight = v
            else:
                print(k)
                assert k == 'not support'

        elif layer_type in ['Detect']:
            indx_in_list = [np.squeeze(np.argwhere(np.asarray(mask_in.cpu().numpy()))) for mask_in in mask_list[layer_idx]]
            if 'm.0.weight' in k:
                tweight = v[:, indx_in_list[0].tolist(), :, :].clone()
            elif 'm.1.weight' in k:
                tweight = v[:, indx_in_list[1].tolist(), :, :].clone()
            elif 'm.2.weight' in k:
                tweight = v[:, indx_in_list[2].tolist(), :, :].clone()
            else:
                tweight = v
        else:
            print(layer_type)
        #print(k, tweight.shape)
        n_state_dict[k] = tweight

    n_model_dict = {}
    n_model_dict['epoch'] = -1
    n_model_dict['best_fitness'] = -1
    n_model_dict['training_results'] = None
    n_model_dict['optimizer'] = None
    n_model_dict['model'] = n_state_dict

    torch.save(n_model_dict, save_path)
    return save_path

def write_config_py(template, save_dir, o_ch_list, n_ch_list, e_list, ein_list, mask_list, len_backbone):
    destination = os.path.join(save_dir, 'pruned_'+os.path.split(template)[-1])
    with open(template) as f:
        model_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    #print(destination)

    for i, n_ch in enumerate(n_ch_list):
        if i < len_backbone - 1:
            layer_type = model_dict['backbone'][i][2]
            args = model_dict['backbone'][i][3]
            if layer_type == 'Focus':
                assert len(args) == 2
                args = [n_ch_list[i][-1], args[1]]
                model_dict['backbone'][i][3] = args
            elif layer_type == 'Conv':
                assert len(args) == 3
                args = [n_ch_list[i][-1], args[1], args[2]]
                model_dict['backbone'][i][3] = args
            elif layer_type == 'C3':
                assert len(args) == 5
                args = [n_ch_list[i][-1], True, 1, [round(e,10) for e in e_list[i]], [round(e,10) for e in ein_list[i]]]  # default: c2, shortcut=True, g=1, e=0.5
                model_dict['backbone'][i][3] = args
            elif layer_type == 'SPP':
                assert len(args) == 3
                args = [n_ch_list[i][-1], args[1], round(e_list[i],6)]
                model_dict['backbone'][i][3] = args
            else:
                assert layer_type == 'not support'
        else:
            layer_type, args = None, None
            if i == len_backbone - 1:
                layer_type = model_dict['backbone'][i][2]
                args = model_dict['backbone'][i][3]
            else:
                layer_type = model_dict['head'][i-len_backbone][2]
                args = model_dict['head'][i-len_backbone][3]

            if layer_type == 'Conv':
                assert len(args) == 3
                args = [n_ch_list[i][-1], args[1], args[2]]
                model_dict['head'][i-len_backbone][3] = args
            elif layer_type == 'C3':
                assert len(args) == 5
                args = [n_ch_list[i][-1], False, 1, [round(e,10) for e in e_list[i]], [round(e,10) for e in ein_list[i]]]  # default: c2, shortcut=True, g=1, e=0.5
                if i == len_backbone - 1:
                    model_dict['backbone'][i][3] = args
                else:
                    model_dict['head'][i-len_backbone][3] = args
            elif layer_type in ['nn.Upsample', 'Concat', 'Detect']:
                continue
            else:
                assert layer_type == 'not support'
    #print(model_dict)
    with open(destination, 'w') as ff:
        #yaml.dump(model_dict, ff, sort_keys=False)
        for k, v in model_dict.items():
            ff.write("%s: " % k)
            ff.write(str(v).replace('\'', ' ').replace(' nearest ', '\'nearest\''))
            ff.write('\n')
    return destination

def prune(opt, device):
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = int(data_dict['nc']), data_dict['names']  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    model = Model(opt.cfg, nc=nc).to(device)
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    if opt.weights.endswith('.pt'):  # pytorch format
        ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
        # load model
        try:
           exclude = []  # exclude keys
           model_items = None
           if isinstance(ckpt['model'], Model):
               model_items = ckpt['model'].float().state_dict().items()
           else:
               model_items = ckpt['model'].items()
           model_items = {k: v for k, v in model_items  if k in model.state_dict() and not any(x in k for x in exclude)}
           model.load_state_dict(model_items, strict=True)
           print('Transferred %g/%g items from %s' % (len(model_items), len(model.state_dict()), opt.weights))
           # The weight of the model to be pruned should be strictly corresponding to the model
           #ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
           #                 if k in model.state_dict() and not any(x in k for x in exclude)
           #                 and model.state_dict()[k].shape == v.shape}
           #model.load_state_dict(ckpt['model'], strict=False)
           #print('Transferred %g/%g items from %s' % (len(ckpt['model']), len(model.state_dict()), opt.weights))
        except KeyError as e:
           s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
               "Please delete or update %s and try again, or use --weights '' to train from scratch." \
               % (weights, opt.cfg, opt.weights, opt.weights)
           raise KeyError(s) from e
        del ckpt

    model.eval()

    #print(model)
    net_channel_1 = channel_count_rough(model)
    print("The total number of channels in the model before pruning is ", net_channel_1)

    with open(opt.cfg) as f:
        model_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    backbone_ifo, head_ifo = parse_model(model_dict)
    backbone_ifo_ori = backbone_ifo.copy()
    head_ifo_ori = head_ifo.copy()

    # prune
    save, overall_ratio, perlayer_ratio = opt.save, opt.overall_ratio, opt.perlayer_ratio
    if save != None:
        if not os.path.exists(save):
            os.makedirs(save)

    thresh = grab_thresh(model, overall_ratio, save, opt.weights)
    len_backbone = len(backbone_ifo)
    len_head = len(head_ifo)
    mask_list, o_ch_list, n_ch_list, e_list, ein_list = [], [], [], [], []
    for i, m in enumerate(model.model):
        if i < len_backbone-1:
            # have_shortcut = True
            if isinstance(m, Focus):
                mask = cal_mask_weightingByKernel(thresh, opt.perlayer_ratio, block_conv=m.conv.conv, block_bn=m.conv.bn)
                mask_list.append([mask])
                o_ch_list.append([mask.shape[0]])
                n_ch_list.append([int(torch.sum(mask))])
                e_list.append(-1)
                ein_list.append(-1)
            elif isinstance(m, Conv):
                mask = cal_mask_weightingByKernel(thresh, opt.perlayer_ratio, block_conv=m.conv, block_bn=m.bn)
                mask_list.append([mask])
                o_ch_list.append([mask.shape[0]])
                n_ch_list.append([int(torch.sum(mask))])
                e_list.append(-1)
                ein_list.append(-1)
            elif isinstance(m, C3):
                mask_cv2 = cal_mask_weightingByKernel(thresh, opt.perlayer_ratio, block_conv=m.cv2.conv, block_bn=m.cv2.bn)
                mask_cv3 = cal_mask_weightingByKernel(thresh, opt.perlayer_ratio, block_conv=m.cv3.conv, block_bn=m.cv3.bn)
                c_out = torch.sum(mask_cv3).cpu().numpy()

                block_t = m.m
                len_bottleneck = len(block_t)
                assert backbone_ifo[i][3][1] == True and isinstance(block_t, nn.Sequential) and len_bottleneck == int(backbone_ifo[i][1])

                mask_list1 = [cal_mask_weightingByKernel(thresh, opt.perlayer_ratio, block_conv=block_t[t].cv1.conv, block_bn=block_t[t].cv1.bn) for t in range(len_bottleneck) ]
                mask_list2 = []

                m_cv1_conv_weight, m_cv1_bn_weight = m.cv1.conv.weight.data.abs().clone(), m.cv1.bn.weight.data.abs().clone()
                weight_alpha_list2 = [ torch.mean( block_t[t].cv2.conv.weight.data.abs().clone().view(block_t[t].cv2.conv.weight.data.size()[0], -1), dim=1 )
                                         for t in range(len_bottleneck)]
                weight_copy_list2 = [10 * torch.mean( m_cv1_conv_weight.view(m_cv1_conv_weight.size()[0], -1), dim=1 ) * m_cv1_bn_weight] + [ 10 * weight_alpha_list2[t] * block_t[t].cv2.bn.weight.data.abs().clone() for t in range(len_bottleneck) ]
                weight_copy2 = torch.cat(weight_copy_list2, dim=0)
                mask2 = cal_mask_weight(weight_copy2, thresh, 0.)
                sum2 = torch.sum(mask2).cpu().numpy()
                sum2 = 2*len_bottleneck if sum2 < 2*len_bottleneck else sum2
                och2 = mask2.shape[0]
                rdc2 = int( (och2 - sum2) // (len_bottleneck+1) )
                for weight_copy in weight_copy_list2:
                    mask_t = torch.ones(weight_copy.shape[0])
                    sorted_bn, sorted_index = torch.sort(weight_copy)
                    mask_t[sorted_index[:rdc2]] = 0.
                    mask_list2.append(mask_t.to(device))
                assert torch.sum(mask_list2[0]) == torch.sum(mask_list2[1]) and torch.sum(mask_list2[0]) == torch.sum(mask_list2[-1])
                c_ = torch.sum(mask_list2[0]).cpu().numpy()

                mask_list.append([mask_list1, mask_list2, mask_cv2, mask_cv3])
                o_ch_list.append([ [mask_list1[tt].shape[0] for tt in range(len(mask_list1))], [mask_list2[tt].shape[0] for tt in range(len(mask_list2))], mask_cv2.shape[0], mask_cv3.shape[0] ])
                n_ch_list.append([ [int(torch.sum(mask_list1[tt])) for tt in range(len(mask_list1))], [int(torch.sum(mask_list2[tt])) for tt in range(len(mask_list2))], int(torch.sum(mask_cv2)), int(torch.sum(mask_cv3)) ])
                e_list.append( [torch.sum(mask_list2[0]).cpu().numpy() / c_out, torch.sum(mask_cv2).cpu().numpy() / c_out] )
                ein_list.append( [torch.sum(mask_list1[tt]).cpu().numpy() / c_ for tt in range(len_bottleneck)] )
            elif isinstance(m, SPP):
                mask_cv1 = cal_mask_weightingByKernel(thresh, opt.perlayer_ratio, block_conv = m.cv1.conv, block_bn = m.cv1.bn)
                mask_cv2 = cal_mask_weightingByKernel(thresh, opt.perlayer_ratio, block_conv = m.cv2.conv, block_bn = m.cv2.bn)
                e_list.append(int(torch.sum(mask_cv1)) / n_ch_list[-1][-1])  # MUST First
                mask_list.append([mask_cv1, mask_cv2])
                o_ch_list.append([mask_cv1.shape[0], mask_cv2.shape[0]])
                n_ch_list.append([int(torch.sum(mask_cv1)), int(torch.sum(mask_cv2))])
                ein_list.append(-1)
            else:
                assert type(m) == 'not support'
        else:
            # have_shortcut = False
            if isinstance(m, nn.Upsample):
                mask_list.append(mask_list[-1])
                o_ch_list.append(o_ch_list[-1])
                n_ch_list.append(n_ch_list[-1])
                e_list.append(-1)
                ein_list.append(-1)
            elif isinstance(m, Conv):
                mask = cal_mask_weightingByKernel(thresh, opt.perlayer_ratio, block_conv = m.conv, block_bn = m.bn)
                mask_list.append([mask])
                o_ch_list.append([mask.shape[0]])
                n_ch_list.append([int(torch.sum(mask))])
                e_list.append(-1)
                ein_list.append(-1)
            elif isinstance(m, Concat):
                layer_idx = head_ifo[i-len_backbone][0]
                mask_list_Ccat = [mask_list[idx][-1] for idx in layer_idx]
                mask_list.append( [torch.cat(mask_list_Ccat, dim=0)] )
                o_ch_list.append( [sum([o_ch_list[idx][-1] for idx in layer_idx])] )
                n_ch_list.append( [sum([n_ch_list[idx][-1] for idx in layer_idx])] )
                e_list.append(-1)
                ein_list.append(-1)
            elif isinstance(m, C3):
                mask_cv1 = cal_mask_weightingByKernel(thresh, opt.perlayer_ratio, block_conv = m.cv1.conv, block_bn = m.cv1.bn)
                mask_cv2 = cal_mask_weightingByKernel(thresh, opt.perlayer_ratio, block_conv = m.cv2.conv, block_bn = m.cv2.bn)
                mask_cv3 = cal_mask_weightingByKernel(thresh, opt.perlayer_ratio, block_conv = m.cv3.conv, block_bn = m.cv3.bn)
                c_out = torch.sum(mask_cv3).cpu().numpy()

                block_t = m.m
                len_bottleneck = len(block_t)
                if i < len_backbone:
                    assert backbone_ifo[i][3][1] == False and isinstance(block_t, nn.Sequential) and len_bottleneck == int(backbone_ifo[i][1])
                else:
                    assert head_ifo[i-len_backbone][3][1] == False and isinstance(block_t, nn.Sequential) and len_bottleneck == int(head_ifo[i-len_backbone][1])

                mask_list1 = [ cal_mask_weightingByKernel(thresh, opt.perlayer_ratio, block_conv = block_t[tt].cv1.conv, block_bn = block_t[tt].cv1.bn) for tt in range(len_bottleneck) ]
                mask_list2 = [ cal_mask_weightingByKernel(thresh, opt.perlayer_ratio, block_conv = block_t[tt].cv2.conv, block_bn = block_t[tt].cv2.bn) for tt in range(len_bottleneck) ]
                mask_list2 = [mask_cv1] + mask_list2

                mask_list.append([mask_list1, mask_list2, mask_cv2, mask_cv3])
                o_ch_list.append([ [mask_list1[tt].shape[0] for tt in range(len(mask_list1))], [mask_list2[tt].shape[0] for tt in range(len(mask_list2))], mask_cv2.shape[0], mask_cv3.shape[0] ])
                n_ch_list.append([ [int(torch.sum(mask_list1[tt])) for tt in range(len(mask_list1))], [int(torch.sum(mask_list2[tt])) for tt in range(len(mask_list2))], int(torch.sum(mask_cv2)), int(torch.sum(mask_cv3)) ])
                e_list.append( [torch.sum(mask_list2[0]).cpu().numpy() / c_out, torch.sum(mask_cv2).cpu().numpy() / c_out] )
                ein_list_temp = []
                for tt in range(len_bottleneck):
                    ein_list_temp += [torch.sum(mask_list2[tt]), torch.sum(mask_list1[tt]), torch.sum(mask_list2[tt+1])]
                ein_list_temp = [e.cpu().numpy() / c_out for e in ein_list_temp]
                ein_list.append( ein_list_temp )
            else:    # Detect
                layer_idx = head_ifo[i-len_backbone][0]
                mask_list.append( [mask_list[idx][-1] for idx in layer_idx] )
                o_ch_list.append( [o_ch_list[idx][-1] for idx in layer_idx] )
                n_ch_list.append( [n_ch_list[idx][-1] for idx in layer_idx] )
                e_list.append(-1)
                ein_list.append(-1)

    destination_cfg = write_config_py(opt.cfg, save, o_ch_list, n_ch_list, e_list, ein_list, mask_list, len_backbone)
    destination_pth = extract_weights(opt.cfg, opt.weights, destination_cfg, mask_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', type=int, default=16, help="Total batch size for all gpus.")
    parser.add_argument("--overall_ratio", default=0.5, type=float, help='pruning rate')
    parser.add_argument("--perlayer_ratio", default=0.1, type=float, help='pruning protection rate')
    parser.add_argument("--save", default='prune', type=str, help='path to save pruned model (default: none)')
    opt = parser.parse_args()

    opt.save += "_{}_{}".format(opt.overall_ratio, opt.perlayer_ratio)
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    device = select_device(opt.device, batch_size=opt.batch_size)

    print(opt)
    prune(opt, device)
