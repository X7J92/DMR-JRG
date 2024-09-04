from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import sigmoid

import _init_paths
import os
import pprint
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '/home/l/data_1/wmz3/DepNet_ANet_Release')
from  lib.models.loss_w import weakly_supervised_loss
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from lib.core.eval import  eval_predictions, display_results
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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import torch
# def calculate_sentence_rank_accuracy_with_grounding(text_features, video_features, video_ids, input_video_ids,
#                                                     sentences_per_paragraph_tensor, grounding_masks):
#     # Calculate text-video similarity scores
#     similarity_scores = torch.matmul(text_features, video_features.T)
#
#     # Extract core video IDs from the video library
#     core_video_ids = video_ids
#
#     # Adjust input video IDs to match video library IDs
#     true_video_ids = input_video_ids
#
#     # Initialize counters for different IOU levels and rank thresholds
#     rank1_count = [0, 0, 0]  # for IOU=0.3, 0.5, 0.7
#     rank5_count = [0, 0, 0]
#     rank10_count = [0, 0, 0]
#     rank100_count = [0, 0, 0]
#     total_sentences = sentences_per_paragraph_tensor.sum().item()  # Convert tensor sum to scalar
#
#     sentence_index = 0  # Track sentence index across paragraphs
#
#     for i, true_id in enumerate(true_video_ids):
#         # Get similarity scores and sort indices in descending order
#         sorted_indices = torch.argsort(similarity_scores[i], descending=True)
#         sorted_video_ids = [core_video_ids[idx] for idx in sorted_indices]
#
#         # Get the number of sentences in the current paragraph and ensure it's an integer
#         sentence_count = int(sentences_per_paragraph_tensor[i].item())
#
#         # Check if the correct video is within the top 1, 5, 10, or 100 and adjust counts based on grounding accuracy
#         top_1 = sorted_video_ids[:1]
#         top_5 = sorted_video_ids[:5]
#         top_10 = sorted_video_ids[:10]
#         top_100 = sorted_video_ids[:100]
#
#         for j in range(sentence_count):
#             for k in range(3):  # Check each IOU level
#                 if true_id in top_1 and grounding_masks[k, 0, sentence_index + j]:
#                     rank1_count[k] += 1
#                 if true_id in top_5 and grounding_masks[k, 0, sentence_index + j]:
#                     rank5_count[k] += 1
#                 if true_id in top_10 and grounding_masks[k, 0, sentence_index + j]:
#                     rank10_count[k] += 1
#                 if true_id in top_100 and grounding_masks[k, 0, sentence_index + j]:
#                     rank100_count[k] += 1
#
#         sentence_index += sentence_count  # Update sentence index for next iteration
#
#     # Calculate accuracy for each IOU level at each rank threshold, convert to percentage
#     rank1_accuracy = [(count / total_sentences) * 100 for count in rank1_count]  # Multiply by 100 for percentage
#     rank5_accuracy = [(count / total_sentences) * 100 for count in rank5_count]  # Multiply by 100 for percentage
#     rank10_accuracy = [(count / total_sentences) * 100 for count in rank10_count]  # Multiply by 100 for percentage
#     rank100_accuracy = [(count / total_sentences) * 100 for count in rank100_count]  # Multiply by 100 for percentage
#
#     return rank1_accuracy, rank5_accuracy, rank10_accuracy, rank100_accuracy
import torch

def calculate_sentence_rank_accuracy_with_grounding(text_features, video_features, video_ids, input_video_ids,
                                                    sentences_per_paragraph_tensor, grounding_masks):
    # Calculate text-video similarity scores
    similarity_scores = torch.matmul(text_features, video_features.T)





    # Extract core video IDs from the video library
    core_video_ids = [vid.split('.')[0] for vid in video_ids]

    # Adjust input video IDs to match video library IDs
    true_video_ids = [vid.split('.')[0] for vid in input_video_ids]

    # Initialize counters for different IOU levels and rank thresholds
    rank1_count = [0, 0, 0]  # for IOU=0.3, 0.5, 0.7
    rank5_count = [0, 0, 0]
    rank10_count = [0, 0, 0]
    rank100_count = [0, 0, 0]
    total_sentences = sentences_per_paragraph_tensor.sum().item()  # Convert tensor sum to scalar

    sentence_index = 0  # Track sentence index across paragraphs

    for i, true_id in enumerate(true_video_ids):
        # Get similarity scores and sort indices in descending order
        sorted_indices = torch.argsort(similarity_scores[i], descending=True)
        sorted_video_ids = [core_video_ids[idx] for idx in sorted_indices]

        # Get the number of sentences in the current paragraph and ensure it's an integer
        sentence_count = int(sentences_per_paragraph_tensor[i].item())

        # Determine if the paragraph's retrieval was correct
        retrieval_correct = true_id in sorted_video_ids[:100]  # Assuming Rank-100 is the largest context we consider

        for j in range(sentence_count):
            # Check each IOU level
            for k in range(3):
                if retrieval_correct and true_id in sorted_video_ids[:1] and grounding_masks[k, 0, sentence_index + j]:
                    rank1_count[k] += 1
                if retrieval_correct and true_id in sorted_video_ids[:5] and grounding_masks[k, 0, sentence_index + j]:
                    rank5_count[k] += 1
                if retrieval_correct and true_id in sorted_video_ids[:10] and grounding_masks[k, 0, sentence_index + j]:
                    rank10_count[k] += 1
                if retrieval_correct and true_id in sorted_video_ids[:100] and grounding_masks[k, 0, sentence_index + j]:
                    rank100_count[k] += 1

        sentence_index += sentence_count  # Update sentence index for next iteration

    # Calculate accuracy for each IOU level at each rank threshold, convert to percentage
    rank1_accuracy = [(count / total_sentences) * 100 for count in rank1_count]  # Multiply by 100 for percentage
    rank5_accuracy = [(count / total_sentences) * 100 for count in rank5_count]
    rank10_accuracy = [(count / total_sentences) * 100 for count in rank10_count]
    rank100_accuracy = [(count / total_sentences) * 100 for count in rank100_count]

    return rank1_accuracy, rank5_accuracy, rank10_accuracy, rank100_accuracy

def calculate_sentence_rank_accuracy(text_features, video_features, video_ids, input_video_ids, sentences_per_paragraph_tensor):
    # 计算文本和视频之间的相似度分数
    similarity_scores = torch.matmul(text_features, video_features.T)

    # 从视频库文件名中提取核心视频ID
    core_video_ids = video_ids

    # 从输入的视频ID中移除_后面的部分以匹配视频库ID
    true_video_ids = input_video_ids

    # 初始化计数器
    rank1_count = 0
    rank5_count = 0
    rank10_count = 0
    rank100_count = 0
    total_sentences = sentences_per_paragraph_tensor.sum().item()  # 将张量总和转换为标量

    for i, true_id in enumerate(true_video_ids):
        # 获取相似度分数，排序，找到最高分数的视频索引
        sorted_indices = torch.argsort(similarity_scores[i], descending=True)
        sorted_video_ids = [core_video_ids[idx] for idx in sorted_indices]

        # 句子数量从张量中获取
        sentence_count = sentences_per_paragraph_tensor[i].item()

        # 检查正确的视频是否在排序中的前1, 5, 10, 100位
        if true_id in sorted_video_ids[:1]:
            rank1_count += sentence_count
        if true_id in sorted_video_ids[:5]:
            rank5_count += sentence_count
        if true_id in sorted_video_ids[:10]:
            rank10_count += sentence_count
        if true_id in sorted_video_ids[:100]:
            rank100_count += sentence_count

    rank1_accuracy = rank1_count / total_sentences
    rank5_accuracy = rank5_count / total_sentences
    rank10_accuracy = rank10_count / total_sentences
    rank100_accuracy = rank100_count / total_sentences

    return rank1_accuracy, rank5_accuracy, rank10_accuracy, rank100_accuracy




import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', default='/home/l/data_1/wmz3/DepNet_ANet_Release/experiments/dense_activitynet/acnet.yaml',required=False, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus', type=str)
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

    test_dataset = getattr(datasets, dataset_name)('test')

    model = getattr(models, model_name)()

    model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
    model.load_state_dict(model_checkpoint,strict=True)


    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(test_dataset,
                            batch_size=config.TEST.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.WORKERS,
                            pin_memory=False,
                            collate_fn=datasets.dense_collate_fn)


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
        video_idxs = sample['batch_video_idxs']

        # print(video_idxs)
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

        #prediction (bsz 8 1 32 32)
        # map_mask (bsz 8 1 32 32)
        joint_prob = torch.sigmoid(prediction) * map_mask
        rewards = torch.from_numpy(np.asarray([0, 0.5, 1.0])).cuda()
        # loss_value1 =bce_rescale_loss(prediction, map_mask, sentence_mask,overlaps_tensor_z.unsqueeze(2),config.LOSS.PARAMS)
        loss_value2 = bce_rescale_loss(prediction, map_mask, sentence_mask, overlaps_tensor_f.unsqueeze(2),
                                       config.LOSS.PARAMS)

        loss_w = weakly_supervised_loss(weight_3, words_logit, ids_list,words_mask1,rewards,sentence_mask)

        loss_clip = loss.clip_loss(sims,logit_scale)

        loss_value = loss_value2 +loss_w + loss_clip

        # vg_hs_video, vg_hs_t, vg_hs_v

        sorted_times = None if model.training else get_proposal_results(joint_prob, duration)
        vg_hs_video1= None if model.training else vg_hs_video
        vg_hs_t1 = None if model.training else vg_hs_t
        vg_hs_v1=None if model.training else vg_hs_v


        return loss_value, sorted_times, vg_hs_video1,vg_hs_t1,vg_hs_v1,sentence_number, video_idxs


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



    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list']=[]
        state['sorted_video_list'] = []
        state['sorted_text_list'] = []
        state['sorted_video_l_list'] = []
        state['sentence_number_list'] = []
        state['sorted_video_id_list'] = []
        if config.VERBOSE:
            if state['split'] == 'test':
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
        sorted_videos= state['video'].cpu().detach()
        sorted_texts=state['text'].cpu().detach()
        sorted_videos_l = state['video_l'].cpu().detach()
        sorted_sentence_number=state['sentence_number'].cpu().detach()
        sorted_video_idxs = state['video_idxs']

        # 假设state字典已经被初始化，并且有sorted_video_list和sorted_text_list键
        # state = {'sorted_video_list': [], 'sorted_text_list': []}

        for batch_index in range(sorted_videos.shape[0]):  # 遍历每个批次
            # 从每个批次中提取单个512维向量
            video_vector = sorted_videos[batch_index]
            text_vector = sorted_texts[batch_index]
            video_l_vector=sorted_videos_l[batch_index]
            sorted_sentence_number_vector=sorted_sentence_number[batch_index]
            sorted_video_idxs1= sorted_video_idxs[batch_index]
            # 将这个向量追加到相应的列表中
            state['sorted_video_list'].append(video_vector)
            state['sorted_text_list'].append(text_vector)
            state['sorted_video_l_list'].append(video_l_vector)
            state['sentence_number_list'].append(sorted_sentence_number_vector)
            state['sorted_video_id_list'].append(sorted_video_idxs1)
    def on_test_end(state):
        #############################################################################定位评价指标###################################################################################
        annotations = state['iterator'].dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'], grounding_mask = eval.eval_predictions(state['sorted_segments_list'],
                                                                                      annotations, verbose=False)
        print(
            '################################################################################## 检索评价指标############################################################################')
        sorted_text_list_all = state['sorted_text_list']
        sorted_video_list_all = state['sorted_video_list']
        sorted_video_l_list_all = state['sorted_video_l_list']
        sorted_sentence_number_list_all = state['sentence_number_list']
        sorted_video_id_list_all = state['sorted_video_id_list']
        # print(sorted_video_id_list_all)



        sorted_video_l_list_tensor = torch.stack(sorted_video_l_list_all, dim=0)
        sorted_text_list_tensor = torch.stack(sorted_text_list_all, dim=0)
        sorted_video_list_tensor = torch.stack(sorted_video_list_all, dim=0)
        sorted_sentence_number_list_tensor = torch.stack(sorted_sentence_number_list_all, dim=0)
##############################################################################################################################
        sorted_video_list_tensor_25= sorted_video_list_tensor
        sorted_video_id_list_all_25=sorted_video_id_list_all
        rank1, rank5,rank10, rank100 = calculate_sentence_rank_accuracy(sorted_text_list_tensor, sorted_video_list_tensor_25, sorted_video_id_list_all_25, sorted_video_id_list_all,
                                                           sorted_sentence_number_list_tensor)
        print(f'Rank-1 Accuracy: {rank1:.2f}')
        print(f'Rank-5 Accuracy: {rank5:.2f}')
        print(f'Rank-10 Accuracy: {rank10:.2f}')
        print(f'Rank-100 Accuracy: {rank100:.2f}')
###########################################################################################################################
        print(
            '################################################################################## 检索加定位评价指标############################################################################')

        # 提取前100个特征
        texts = sorted_text_list_tensor[:100]
        videos = sorted_video_list_tensor_25[:100]

        # 合并这些特征以用于t-SNE
        combined_features = np.vstack([texts, videos])

        # 运行t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(combined_features)

        # 可视化
        fig, ax = plt.subplots()

        # 生成100种不同的颜色
        colors = [plt.cm.hsv(i / 100) for i in range(100)]

        # 绘制文本特征
        ax.scatter(reduced_features[:100, 0], reduced_features[:100, 1], c=colors, marker='s', s=50, label='Texts')

        # 绘制视频特征
        ax.scatter(reduced_features[100:, 0], reduced_features[100:, 1], c=colors, marker='^', s=50, label='Videos')

        # 添加图例（可选）
        ax.legend()

        # 保存图像
        plt.savefig('/home/l/data_2/wmz/1_c/DepNet_ANet_Release/tsne_visualization.png')






        rank1_acc, rank5_acc,rank10_acc, rank100_acc = calculate_sentence_rank_accuracy_with_grounding(sorted_text_list_tensor, sorted_video_list_tensor_25, sorted_video_id_list_all_25, sorted_video_id_list_all,
                                                           sorted_sentence_number_list_tensor,grounding_mask)
        print(f'Rank-1 Accuracy at IOU=0.3, 0.5, 0.7: {rank1_acc}')
        print(f'Rank-5 Accuracy at IOU=0.3, 0.5, 0.7: {rank5_acc}')
        print(f'Rank-10 Accuracy at IOU=0.3, 0.5, 0.7: {rank10_acc}')
        print(f'Rank-100 Accuracy at IOU=0.3, 0.5, 0.7: {rank100_acc}')



        if config.VERBOSE:
            state['progress_bar'].close()

    engine = Engine()
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.test(network, dataloader,split='test')


