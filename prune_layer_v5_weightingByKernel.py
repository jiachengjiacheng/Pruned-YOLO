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

def grab_thresh(model, overall_ratio):
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
    return thresh

def parse_model(d):
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    len_backbone = len(d['backbone'])

    grab_ifo_layer_idx, grab_ifo_layer_num = [], []
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
    #for i, (f, n, m, args) in enumerate(d['backbone']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if n > 1 and m in [C3]:
            grab_ifo_layer_idx.append(i)
            grab_ifo_layer_num.append(n)
            #grab_ifo.append({i:n})
    return grab_ifo_layer_idx, grab_ifo_layer_num

def extract_weights(weights, destination, grab_ifo_layer_idx, grab_ifo_layer_num_ori, grab_ifo_layer_num, index_list):
    index_list.sort()
    save_path = destination.replace('.yaml', '.pt')
    print(save_path)
    print(grab_ifo_layer_idx, grab_ifo_layer_num_ori, grab_ifo_layer_num, index_list)

    idx = 0
    o_state_dict = torch.load(weights, map_location=lambda storage, loc: storage)
    for i, (m,n) in enumerate(zip(grab_ifo_layer_num_ori, grab_ifo_layer_num)):
        if m == n:
            continue
        else:
            idx_tlist = index_list[idx: idx+m-n]
            idx_loc_list = []
            for idx_t in idx_tlist:
                idx_loc_list.append( idx_t - sum(grab_ifo_layer_num_ori[:i]) )
            idx += m-n
            n_module_list = []
            for module_idx in range(m):
                if module_idx not in idx_loc_list:
                    n_module_list.append(o_state_dict['model'].model[grab_ifo_layer_idx[i]].m[module_idx])
            n_state_dict = nn.Sequential(*n_module_list)
            o_state_dict['model'].model[grab_ifo_layer_idx[i]].m = n_state_dict

    torch.save(o_state_dict, save_path)
    return save_path

def write_config_py(template, save_dir, grab_ifo_layer_idx, grab_ifo_layer_num):
    destination = os.path.join(save_dir, 'pruned_'+os.path.split(template)[-1])
    with open(template) as f:
        model_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    for j, i in enumerate(grab_ifo_layer_idx):
        if i <= len(model_dict['backbone']) - 1:
            model_dict['backbone'][i][1] = grab_ifo_layer_num[j]
        else:
            model_dict['head'][i-len(model_dict['backbone'])][1] = grab_ifo_layer_num[j]
        if i < len(model_dict['backbone']) - 1:
            model_dict['backbone'][i][-1][-1] = model_dict['backbone'][i][-1][-1][:grab_ifo_layer_num[j]]
        elif i == len(model_dict['backbone']) - 1:
            if grab_ifo_layer_num[j] == 0:
                model_dict['backbone'][i][-1][-1] = [model_dict['backbone'][i][-1][-1][-1]]
            else:
                model_dict['backbone'][i][-1][-1] = model_dict['backbone'][i][-1][-1][:3*grab_ifo_layer_num[j]]
        else:
            if grab_ifo_layer_num[j] == 0:
                model_dict['head'][i - len(model_dict['backbone'])][-1][-1] = [model_dict['head'][i - len(model_dict['backbone'])][-1][-1][-1]]
            else:
                model_dict['head'][i-len(model_dict['backbone'])][-1][-1] = model_dict['head'][i-len(model_dict['backbone'])][-1][-1][:3*grab_ifo_layer_num[j]]
    print(destination)
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
           ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                            if k in model.state_dict() and not any(x in k for x in exclude)
                            and model.state_dict()[k].shape == v.shape}
           model.load_state_dict(ckpt['model'], strict=True)
           print('Transferred %g/%g items from %s' % (len(ckpt['model']), len(model.state_dict()), opt.weights))
        except KeyError as e:
           s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
               "Please delete or update %s and try again, or use --weights '' to train from scratch." \
               % (weights, opt.cfg, opt.weights, opt.weights)
           raise KeyError(s) from e
        del ckpt

    #print(model)
    net_channel_1 = channel_count_rough(model)
    print("The total number of channels in the model before pruning is ", net_channel_1)

    with open(opt.cfg) as f:
        model_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    grab_ifo_layer_idx, grab_ifo_layer_num = parse_model(model_dict)
    grab_ifo_layer_num_ori = grab_ifo_layer_num.copy()

    # prune
    save, overall_ratio = opt.save, 0.5
    if save != None:
        if not os.path.exists(save):
            os.makedirs(save)

    thresh = grab_thresh(model, overall_ratio)

    bn_mean_list = []
    bn_mean_chan = []
    for i, m in enumerate(model.model):
        if i in grab_ifo_layer_idx:
            m = m.m
            for j, n in enumerate(m):
                conv_copy = n.cv2.conv.state_dict()['weight'].abs().clone().cpu()
                weight_alpha = torch.mean( conv_copy.view(conv_copy.size()[0], -1), dim=1 )
                bn_weight = n.cv2.bn.state_dict()['weight'].abs().clone().cpu()
                assert weight_alpha.size() == bn_weight.size()
                weight_copy = 10 * weight_alpha * bn_weight

                bn_mean_list.append(torch.mean(weight_copy).numpy().tolist())
                bn_mean_chan.append(weight_copy.numpy().size)

    index_list = [i[0] for i in sorted(enumerate(bn_mean_list), key=lambda x:x[1])]

    for t in range(opt.overall_layers):
        if index_list[t] < grab_ifo_layer_num_ori[0]:
            grab_ifo_layer_num[0] -= 1
        elif index_list[t] < sum(grab_ifo_layer_num_ori[:2]):
            grab_ifo_layer_num[1] -= 1
        elif index_list[t] < sum(grab_ifo_layer_num_ori[:3]):
            grab_ifo_layer_num[2] -= 1
        elif index_list[t] < sum(grab_ifo_layer_num_ori[:4]):
            grab_ifo_layer_num[3] -= 1
        elif index_list[t] < sum(grab_ifo_layer_num_ori[:5]):
            grab_ifo_layer_num[4] -= 1
        elif index_list[t] < sum(grab_ifo_layer_num_ori[:6]):
            grab_ifo_layer_num[5] -= 1
        elif index_list[t] < sum(grab_ifo_layer_num_ori[:7]):
            grab_ifo_layer_num[6] -= 1
        elif index_list[t] < sum(grab_ifo_layer_num_ori[:8]):
            grab_ifo_layer_num[7] -= 1
        else:
            assert 'Not support' == 'Out of range'

    destination_cfg = write_config_py(opt.cfg, save, grab_ifo_layer_idx, grab_ifo_layer_num)
    destination_pth = extract_weights(opt.weights, destination_cfg, grab_ifo_layer_idx, grab_ifo_layer_num_ori, grab_ifo_layer_num, index_list[:opt.overall_layers])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', type=int, default=16, help="Total batch size for all gpus.")
    parser.add_argument("--save", default='prune', type=str, help='path to save pruned model (default: none)')
    parser.add_argument("--overall_layers", default=3, type=int, help='pruning layers')
    opt = parser.parse_args()

    opt.save += "_{}".format(opt.overall_layers)
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    device = select_device(opt.device, batch_size=opt.batch_size)

    print(opt)
    prune(opt, device)
