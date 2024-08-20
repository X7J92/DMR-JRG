from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import _init_paths
import os
import pprint
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '/home/l/data_2/wmz/1_c/DepNet_ANet_Release')
from  lib.models.loss_w import weakly_supervised_loss
from lib.models.rec_t import weakly_supervised_loss_text
import torch.optim as optim
from tqdm import tqdm
from lib.models.loss_t import get_frame_trip_loss,calculate_mse_loss
from lib import datasets
from lib import models
from lib.core.config import config, update_config
from lib.core.engine import Engine
from lib.core.utils import AverageMeter
from lib.core import eval
from lib.core.utils import create_logger
import lib.models.loss as loss
import math
import pickle
from IPython import embed
from  lib.models.loss import bce_rescale_loss
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn as nn

def calculate_sentence_ranks_grounding(similarity_matrix, num_sentences_per_paragraph, iou_masks):
    # 段落级别的排名计算
    paragraph_ranks = []
    num_paragraphs = similarity_matrix.shape[0]

    for i in range(num_paragraphs):
        sims = similarity_matrix[i]
        sorted_indices = torch.argsort(sims, descending=True)
        target_index = sorted_indices == i
        rank = torch.where(target_index)[0] + 1
        num_sentences = int(num_sentences_per_paragraph[i].item())  # 将其转换为整数
        paragraph_ranks.extend([rank.item()] * num_sentences)  # 确保乘法操作正确


    # 生成总句子级排名列表
    ranks = torch.tensor(paragraph_ranks)

    # 创建DataFrame来保存计算结果
    results = {}

    # 定义IOU阈值列表
    iou_thresholds = [0.3, 0.5, 0.7]
    rank_levels = [1, 5, 10, 100]

    # 总句子数
    total_sentences = int(num_sentences_per_paragraph.sum().item())

    for rank_level in rank_levels:
        results[f"Rank@{rank_level}"] = {}
        for idx, iou in enumerate(iou_thresholds):
            iou_mask = iou_masks[idx, 0, :total_sentences]
            valid_ranks = ranks[iou_mask]
            results[f"Rank@{rank_level}"][f"IOU={iou}"] = (torch.sum(
                valid_ranks <= rank_level).item() / ranks.numel()) * 100

    return results



def calculate_sentence_ranks(similarity_matrix, num_sentences_per_paragraph):
    # 段落级别的排名计算
    paragraph_ranks = []
    num_paragraphs = similarity_matrix.shape[0]

    for i in range(num_paragraphs):
        # 获取第i个段落与所有视频的相似度
        sims = similarity_matrix[i]
        # 对相似度进行降序排列并获取索引
        sorted_indices = torch.argsort(sims, descending=True)
        # 找到真正匹配的视频索引
        target_index = sorted_indices == i
        # 获取正确匹配视频的排名
        rank = torch.where(target_index)[0] + 1
        # 段落中的每个句子都被分配相同的排名
        # paragraph_ranks.extend([rank.item()] * num_sentences_per_paragraph[i].item())
        num_sentences = int(num_sentences_per_paragraph[i].item())  # 将其转换为整数
        paragraph_ranks.extend([rank.item()] * num_sentences)  # 确保乘法操作正确


    # 计算句子级Rank-1, Rank-5, Rank-10, Rank-100
    ranks = torch.tensor(paragraph_ranks)
    total_sentences = num_sentences_per_paragraph.sum().item()  # 计算所有句子的总数
    rank1 = torch.sum(ranks <= 1).item() / total_sentences
    rank5 = torch.sum(ranks <= 5).item() / total_sentences
    rank10 = torch.sum(ranks <= 10).item() / total_sentences
    rank100 = torch.sum(ranks <= 100).item() / total_sentences
    return rank1, rank5, rank10, rank100

def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', default='../experiments/dense_charades/charades.yaml',required=False, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', default='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag


if __name__ == '__main__':

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(config, args.cfg, config.TAG)
    logger.info('\n'+pprint.pformat(args))
    logger.info('\n'+pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME

    train_dataset = getattr(datasets, dataset_name)('train')
    if config.TEST.EVAL_TRAIN:
        eval_train_dataset = getattr(datasets, dataset_name)('train')
    if not config.DATASET.NO_VAL:
        val_dataset = getattr(datasets, dataset_name)('val')
    test_dataset = getattr(datasets, dataset_name)('test')

    model = getattr(models, model_name)()
    if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
        model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)

    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=config.TRAIN.LR, betas=(0.9, 0.999), weight_decay=config.TRAIN.WEIGHT_DECAY)
    # optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, momentum=0.9, weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.TRAIN.FACTOR, patience=config.TRAIN.PATIENCE, verbose=config.VERBOSE)



    def iterator(split):
        if split == 'train':
            dataloader = DataLoader(train_dataset,
                                    batch_size=config.TRAIN.BATCH_SIZE,
                                    shuffle=config.TRAIN.SHUFFLE,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.dense_collate_fn)
        elif split == 'val':
            dataloader = DataLoader(val_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.dense_collate_fn)
        elif split == 'test':
            dataloader = DataLoader(test_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.dense_collate_fn)
        elif split == 'train_no_shuffle':
            dataloader = DataLoader(eval_train_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.dense_collate_fn)
        else:
            raise NotImplementedError

        return dataloader



    def network(sample):
        # identical as single
        # anno_idxs:(b,) list
        # visual_input: (b,256,500) tensor

        # different due to dense
        # textual_input: (b,K,seq,300) tensor
        # textual_mask: (b,K,seq,1) tensor
        # sentence_mask: (b,K,1) tensor
        # map_gt: (b,K,1,64,64) tensor

        anno_idxs = sample['batch_anno_idxs']
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        sentence_mask = sample['batch_sentence_mask'].cuda()  # new
        visual_input = sample['batch_vis_input'].cuda()
        map_gt = sample['batch_map_gt'].cuda()
        # torch.Size([4, 8, 1, 32, 32])
        duration = sample['batch_duration']
        weights_list = sample['batch_weights_list']
        ids_list = sample['batch_ids_list']
        ids_list = ids_list.squeeze().int()
        # print(type(ids_list))
        # print(weights_list.shape)torch.Size([4, 8, 24, 1])
        # print(textual_mask.shape)torch.Size([4, 8, 24, 1])

        sentence_mask1=sentence_mask.squeeze()
        sentence_number=sentence_mask1.sum(dim=1).unsqueeze(1)
        prediction, map_mask, sims, logit_scale, sims2, logit_scale2,  sims3, logit_scale3,jj, weight_3, words_logit, ids_list, weights, words_mask1, overlaps_tensor_z, p_values_tensor_z, overlaps_tensor_f, p_values_tensor_f , vg_hs_video,vg_hs_t,vg_hs_v= model(
            textual_input, textual_mask, sentence_mask, visual_input, duration, weights_list, ids_list)

        rewards = torch.from_numpy(np.asarray([0, 0.5, 1.0])).cuda()
        loss_value1 =bce_rescale_loss(prediction, map_mask, sentence_mask,overlaps_tensor_z.unsqueeze(2),config.LOSS.PARAMS)

        loss_value2 = bce_rescale_loss(prediction, map_mask, sentence_mask, overlaps_tensor_f.unsqueeze(2),
                                       config.LOSS.PARAMS)

        # loss_value1, loss_overlap, loss_order, joint_prob = getattr(loss, config.LOSS.NAME)(prediction, map_mask, sentence_mask, overlaps_tensor.unsqueeze(2),config.LOSS.PARAMS)
        joint_prob = torch.sigmoid(prediction) * map_mask
        # loss_w = weakly_supervised_loss(**output, rewards=rewards)
        loss_w = weakly_supervised_loss(weight_3, words_logit, ids_list,words_mask1,rewards,sentence_mask)
        # loss_t = weakly_supervised_loss_text(words_logit, ids_list, words_mask1)
        loss_clip = loss.clip_loss(sims,logit_scale)
        loss_clip2 = loss.clip_loss(sims2, logit_scale2)
        loss_clip3 = loss.clip_loss(sims3, logit_scale3)
        # print(loss_value1)
        # print(loss_value2)
        # loss_value = loss_value2 +0.1*loss_w+ loss_clip
        sims_gt=sims.detach()
        mse_loss = calculate_mse_loss(sims2,sims_gt)
        triplet_loss=get_frame_trip_loss(sims)
        triplet_loss2 = get_frame_trip_loss(sims2)
        triplet_loss3 = get_frame_trip_loss(sims3)
        loss_value = loss_value1+loss_value2 + loss_w + loss_clip+ loss_clip2 + triplet_loss + triplet_loss2+mse_loss+loss_clip3 +triplet_loss3
        # loss_value =  loss_w + + loss_clip
        sorted_times = None if model.training else get_proposal_results(joint_prob, duration)

        vg_hs_video1= None if model.training else vg_hs_video
        vg_hs_t1 = None if model.training else vg_hs_t
        vg_hs_v1=None if model.training else vg_hs_v
        return loss_value, sorted_times,vg_hs_video1,vg_hs_t1,vg_hs_v1,sentence_number

    def get_proposal_results(scores, durations):
        # assume all valid scores are larger than one
        out_sorted_times = []
        for score_sent, duration in zip(scores, durations):
            sent_times = []
            for score in score_sent:
                if score.sum() < 1e-3:
                    break
                T = score.shape[-1]
                sorted_indexs = np.dstack(
                    np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
                sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)

                sorted_indexs[:, 1] = sorted_indexs[:, 1] + 1
                sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
                target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
                sent_times.append((sorted_indexs.float() / target_size * duration).tolist())
            out_sorted_times.append(sent_times)
        return out_sorted_times

    def on_start(state):
        state['loss_meter'] = AverageMeter()
        state['test_interval'] = int(len(train_dataset)/config.TRAIN.BATCH_SIZE*config.TEST.INTERVAL)
        state['t'] = 1
        model.train()
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=state['test_interval'])

    def on_forward(state):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        state['loss_meter'].update(state['loss'].item(), 1)

    def on_update(state):# Save All
        if config.VERBOSE:
            state['progress_bar'].update(1)

        if state['t'] % state['test_interval'] == 0:
            model.eval()
            if config.VERBOSE:
                state['progress_bar'].close()

            loss_message = '\niter: {} train loss {:.4f}'.format(state['t'], state['loss_meter'].avg)
            table_message = ''
            if config.TEST.EVAL_TRAIN:
                train_state = engine.test(network, iterator('train_no_shuffle'), 'train')
                train_table = eval.display_results(train_state['Rank@N,mIoU@M'], train_state['miou'],
                                                   'performance on training set')
                table_message += '\n'+ train_table
            if not config.DATASET.NO_VAL:
                val_state = engine.test(network, iterator('val'), 'val')
                state['scheduler'].step(-val_state['loss_meter'].avg)
                loss_message += ' val loss {:.4f}'.format(val_state['loss_meter'].avg)
                val_state['loss_meter'].reset()
                val_table = eval.display_results(val_state['Rank@N,mIoU@M'], val_state['miou'],
                                                 'performance on validation set')
                table_message += '\n'+ val_table

            test_state = engine.test(network, iterator('test'), 'test')
            loss_message += ' test loss {:.4f}'.format(test_state['loss_meter'].avg)
            test_state['loss_meter'].reset()
            test_table = eval.display_results(test_state['Rank@N,mIoU@M'], test_state['miou'],
                                              'performance on testing set')
            table_message += '\n' + test_table

            message = loss_message+table_message+'\n'
            logger.info(message)


            cfg_name = args.cfg
            cfg_name = os.path.basename(cfg_name).split('.yaml')[0]
            saved_model_filename = os.path.join(config.MODEL_DIR, config.DATASET.NAME, cfg_name)
            if not os.path.exists(saved_model_filename):
                print('Make directory %s ...' % saved_model_filename)
                os.makedirs(saved_model_filename)
            saved_model_filename = os.path.join(
                saved_model_filename,
                'iter{:06d}.pkl'.format(state['t'])
            )

            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), saved_model_filename)
            else:
                torch.save(model.state_dict(), saved_model_filename)


            if config.VERBOSE:
                state['progress_bar'] = tqdm(total=state['test_interval'])
            model.train()
            state['loss_meter'].reset()

    def on_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()


    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        state['sorted_video_list'] = []
        state['sorted_text_list'] = []
        state['sorted_video_l_list'] = []
        state['sentence_number_list'] = []
        if config.VERBOSE:
            if state['split'] == 'train':
                state['progress_bar'] = tqdm(total=math.ceil(len(train_dataset)/config.TEST.BATCH_SIZE))
            elif state['split'] == 'val':
                state['progress_bar'] = tqdm(total=math.ceil(len(val_dataset)/config.TEST.BATCH_SIZE))
            elif state['split'] == 'test':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset)/config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError

    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)
        sorted_videos = state['video'].cpu().detach()
        sorted_texts = state['text'].cpu().detach()
        sorted_videos_l = state['video_l'].cpu().detach()
        sorted_sentence_number=state['sentence_number'].cpu().detach()
        # 假设state字典已经被初始化，并且有sorted_video_list和sorted_text_list键
        # state = {'sorted_video_list': [], 'sorted_text_list': []}

        for batch_index in range(sorted_videos.shape[0]):  # 遍历每个批次
            # 从每个批次中提取单个512维向量
            video_vector = sorted_videos[batch_index]
            text_vector = sorted_texts[batch_index]
            video_l_vector = sorted_videos_l[batch_index]
            sorted_sentence_number_vector = sorted_sentence_number[batch_index]
            # 将这个向量追加到相应的列表中
            state['sorted_video_list'].append(video_vector)
            state['sorted_text_list'].append(text_vector)
            state['sorted_video_l_list'].append(video_l_vector)
            state['sentence_number_list'].append(sorted_sentence_number_vector)
    def on_test_end(state):
        #############################################################################定位评价指标###################################################################################
        annotations = state['iterator'].dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'],grounding_mask = eval.eval_predictions(state['sorted_segments_list'], annotations, verbose=False)
        print('################################################################################## 检索评价指标############################################################################')
        sorted_text_list_all = state['sorted_text_list']
        sorted_video_list_all = state['sorted_video_list']
        sorted_video_l_list_all = state['sorted_video_l_list']
        sorted_sentence_number_list_all = state['sentence_number_list']

        sorted_video_l_list_tensor = torch.stack(sorted_video_l_list_all, dim=0)
        sorted_text_list_tensor = torch.stack(sorted_text_list_all, dim=0)
        sorted_video_list_tensor = torch.stack(sorted_video_list_all, dim=0)
        sorted_sentence_number_list_tensor = torch.stack(sorted_sentence_number_list_all, dim=0)

        sims_l = torch.matmul(sorted_video_l_list_tensor, sorted_text_list_tensor.T)
        rank1, rank5, rank10, rank100 = calculate_sentence_ranks(sims_l, sorted_sentence_number_list_tensor)
        # 将结果以表格形式展示
        results1 = pd.DataFrame({
            "Rank-1": [rank1 * 100],
            "Rank-5": [rank5 * 100],
            "Rank-10": [rank10 * 100],
            "Rank-100": [rank100 * 100]
        })
        print('使用定位增强检索特征进行检索相似度计算')
        print(results1)
        print('---------------------------------------------------------------------------------------------------------')
        sims_g = torch.matmul(sorted_video_list_tensor, sorted_text_list_tensor.T)
        rank1, rank5, rank10, rank100 = calculate_sentence_ranks(sims_g, sorted_sentence_number_list_tensor)
        results2 = pd.DataFrame({
            "Rank-1": [rank1 * 100],
            "Rank-5": [rank5 * 100],
            "Rank-10": [rank10 * 100],
            "Rank-100": [rank100 * 100]
        })
        print('使用全局特征进行检索相似度计算')
        print(results2)

        print('################################################################################## 检索加定位评价指标###########################################################################')
        r_l = calculate_sentence_ranks_grounding(sims_l, sorted_sentence_number_list_tensor, grounding_mask)
        print('使用定位增强检索特征进行检索定位')
        for rank, data in r_l.items():
            df = pd.DataFrame(data, index=[rank])
            print(df)
            print("\n" + "=" * 40 + "\n")  # 添加分隔线以区分不同的表格

        print('---------------------------------------------------------------------------------------------------------')

        r_g = calculate_sentence_ranks_grounding(sims_g, sorted_sentence_number_list_tensor, grounding_mask)
        print('使用全局特征进行检索定位')
        for rank, data in r_g.items():
            df = pd.DataFrame(data, index=[rank])
            print(df)
            print("\n" + "=" * 40 + "\n")  # 添加分隔线以区分不同的表格

        if config.VERBOSE:
            state['progress_bar'].close()



    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.train(network,
                 iterator('train'),
                 maxepoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler)