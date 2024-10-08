import json
import argparse
import numpy as np
from terminaltables import AsciiTable
import torch
from lib.core.config import config, update_config

from IPython import embed
def list_dimensions(lst):
    if isinstance(lst, list):
        if lst:
            return [len(lst)] + list_dimensions(lst[0])
        return [0]
    return []
def iou(pred, gt): # require pred and gt is numpy
    assert isinstance(pred, list) and isinstance(gt,list)
    pred_is_list = isinstance(pred[0],list)
    gt_is_list = isinstance(gt[0],list)
    if not pred_is_list: pred = [pred]
    if not gt_is_list: gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:,0,None], gt[None,:,0])
    inter_right = np.minimum(pred[:,1,None], gt[None,:,1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:,0,None], gt[None,:,0])
    union_right = np.maximum(pred[:,1,None], gt[None,:,1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:,0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap

def rank(pred, gt):
    return pred.index(gt) + 1

def nms(dets, thresh=0.4, top_k=-1):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    order = np.arange(0,len(dets),1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]

def eval(segments, data):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU,str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL,str) else [config.TEST.RECALL]

    eval_result = [[[] for _ in recalls] for _ in tious]
    max_recall = max(recalls)
    average_iou = []
    count = 0
    for segs, dats in zip(segments, data):
        # assert len(segs) == len(dats['timestamps']) or len(segs) == 8
        if not (len(segs) == len(dats['timestamps']) or len(segs) == 8):
            count += 1
            print(count)
            if (len(segs) != 0):
                print(len(segs))
            continue
        for seg, dat in zip(segs, dats['timestamps'][:len(segs)]):
            seg = nms(seg, thresh=config.TEST.NMS_THRESH, top_k=max_recall).tolist()
#             print(seg)[[57.29875183105469, 79.72000122070312], [69.7550048828125, 79.72000122070312], [74.73750305175781, 79.72000122070312], [39.86000061035156, 77.22875213623047], [77.22875213623047, 79.72000122070312]]
# [[0.0, 171.84375], [0.0, 40.09687423706055], [0.0, 80.1937484741211], [0.0, 17.184375762939453], [28.640625, 114.5625]]
            overlap = iou(seg, [dat])
            average_iou.append(np.mean(np.sort(overlap[0])[-3:]))

            for i,t in enumerate(tious):
                for j,r in enumerate(recalls):
                    eval_result[i][j].append((overlap > t)[:r].any())
    # print(eval_result.shape)

    # eval_result_tensor = torch.stack(eval_result, dim=0)
    # print(list_dimensions(eval_result))
    eval_result_tensor = torch.tensor(eval_result)
    # print(eval_result_tensor.shape)
    eval_result = np.array(eval_result).mean(axis=-1)
    miou = np.mean(average_iou)
    grounding_mask=eval_result_tensor
    return eval_result, miou,grounding_mask

def eval_predictions(segments, data, verbose=True):
    eval_result, miou,grounding_mask = eval(segments, data)
    if verbose:
        print(display_results(eval_result, miou, ''))

    return eval_result, miou,grounding_mask

def display_results(eval_result, miou, title=None):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU,str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL,str) else [config.TEST.RECALL]

    display_data = [['Rank@{},mIoU@{}'.format(i,j) for i in recalls for j in tious]+['mIoU']]
    eval_result = eval_result*100
    miou = miou*100
    display_data.append(['{:.02f}'.format(eval_result[j][i]) for i in range(len(recalls)) for j in range(len(tious))]
                        +['{:.02f}'.format(miou)])
    table = AsciiTable(display_data, title)
    for i in range(len(tious)*len(recalls)):
        table.justify_columns[i] = 'center'
    return table.table


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.verbose:
        config.VERBOSE = args.verbose

if __name__ == '__main__':
    args = parse_args()
    reset_config(config, args)
    train_data = json.load(open('/data/home2/hacker01/Data/DiDeMo/train_data.json', 'r'))
    val_data = json.load(open('/data/home2/hacker01/Data/DiDeMo/val_data.json', 'r'))

    moment_frequency_dict = {}
    for d in train_data:
        times = [t for t in d['times']]
        for time in times:
            time = tuple(time)
            if time not in moment_frequency_dict.keys():
                moment_frequency_dict[time] = 0
            moment_frequency_dict[time] += 1

    prior = sorted(moment_frequency_dict, key=moment_frequency_dict.get, reverse=True)
    prior = [list(item) for item in prior]
    prediction = [prior for d in val_data]

    eval_predictions(prediction, val_data)