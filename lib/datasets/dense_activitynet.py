# """ Dataset loader for the ActivityNet Captions dataset """
# import os
# import json
#
# import h5py
# import torch
# from torch import nn
# import torch.nn.functional as F
# import torch.utils.data as data
# import torchtext
#
# from . import average_to_fixed_length
# from lib.core.eval import iou
# from lib.core.config import config
# from torch.nn.utils.rnn import pad_sequence
#
# import numpy as np
# import random
# from IPython import embed
# import nltk
# import pickle
#
# with open('/home/pc/data/wmz/2/DepNet_ANet_Release/data/ActivityNet/vocab.pkl', 'rb') as fp:
#     vocabs = pickle.load(fp)
#
# class DenseActivityNet(data.Dataset):
#
#     vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"](cache='../../data/glove')
#     vocab.itos.extend(['<unk>'])
#     vocab.stoi['<unk>'] = vocab.vectors.shape[0]
#     vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
#     word_embedding = nn.Embedding.from_pretrained(vocab.vectors)
#
#
#
#     def __init__(self, split):
#         super(DenseActivityNet, self).__init__()
#
#         self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
#         self.data_dir = config.DATA_DIR
#         self.split = split
#         self.vocabs = vocabs
#
#         self.keep_vocab = dict()
#         for w, _ in vocabs['counter'].most_common(8000):
#             self.keep_vocab[w] = self.vocab_size
#
#         # val_1.json is renamed as val.json, val_2.json is renamed as test.json
#         if split == 'test':
#             self.annotations = np.load(os.path.join('/home/pc/data/wmz/2/DepNet_ANet_Release/data/ActivityNet/' '{}.npy'.format(split)),  allow_pickle=True)
#         else:
#             with open(os.path.join(self.data_dir, '{}.json'.format(split)),'r') as f:
#                 annotations = json.load(f)
#             anno_pairs = []
#             for vid, video_anno in annotations.items():
#                 duration = video_anno['duration']
#                 flag = True
#                 for timestamp in video_anno['timestamps']:
#                     if timestamp[0] >= timestamp[1]:
#                         flag = False
#                 if not flag:
#                     continue
#                 anno_pairs.append(
#                     {
#                         'video': vid,
#                         'duration': duration,
#                         'sentences': video_anno['sentences'],
#                         'timestamps': video_anno['timestamps']
#                     }
#                 )
#             self.annotations = anno_pairs
#
#     def vocab_size(self):
#         return len(self.keep_vocab) + 1
#
#     def __getitem__(self, index):
#         # 从annotations中获取视频ID和时长
#         video_id = self.annotations[index]['video']
#         duration = self.annotations[index]['duration']
#
#         # 获取与视频相关的句子总数
#         tot_sentence = len(self.annotations[index]['sentences'])
#
#         # 使用所有句子的索引
#         idx_sample = list(range(tot_sentence))
#         idx_sample.sort()  # 这一步实际上是多余的，因为range已经是排序的
#
#         # 获取所有句子及其时间戳
#         sentence_sample = [self.annotations[index]['sentences'][idx] for idx in idx_sample]
#         timestamps_sample = [self.annotations[index]['timestamps'][idx] for idx in idx_sample]
#
#         # word_vectors_list 和 txt_mask_list 的处理逻辑未提供，需要根据您的需求定制
#         # print(sentence_sample)
#         word_vectors_list = []
#         txt_mask_list = []
#         weights_list = []
#         # for sentence in self.annotations[index]['sentences']:
#         # print(sentence_sample)
#         words=[]
#         word_id_list= []
#         for sentence in sentence_sample:
#             # print(sentence)
#
#
#
#
#             word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()],
#                                      dtype=torch.long)
#             word_vectors = self.word_embedding(word_idxs) # word_vectors (seq, 300)
#             word_vectors_list.append(word_vectors)
#             txt_mask_list.append(torch.ones(word_vectors.shape[0], 1))
#             # 处理每个句子的权重
#             sentence_weights = []
#             for tag in nltk.pos_tag([word.lower() for word in sentence.split()]):
#                 if 'NN' in tag:
#                     sentence_weights.append(2)
#                 elif 'VB' in tag:
#                     sentence_weights.append(2)
#                 elif 'JJ' in tag or 'RB' in tag:
#                     sentence_weights.append(2)
#                 else:
#                     sentence_weights.append(1)
#
#
#
#
#
#
#             word_id_list.append(word_idxs)
#             weights_list.append(sentence_weights)
#
#         word_vectors_list = pad_sequence(word_vectors_list, batch_first=True) # word_vectors_list (k, seq, 300)
#
#         txt_mask_list = pad_sequence(txt_mask_list, batch_first=True) # txt_mask_list (k, seq, 1)
#
#         word_id_list = [subl.clone().detach().unsqueeze(1) for subl in word_id_list]
#         word_id_list = pad_sequence(word_id_list,batch_first=True)
#         print(word_id_list.shape)
#
#         weights_list = [torch.tensor(sublist).unsqueeze(1) for sublist in weights_list]
#         weights_list = pad_sequence(weights_list, batch_first=True)
#         visual_input, visual_mask = self.get_video_features(video_id)
#
#         # Time scaled to same size
#         if config.DATASET.NUM_SAMPLE_CLIPS > 0:
#             visual_input = average_to_fixed_length(visual_input)
#             num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
#
#             overlaps_list = []
#             # for gt_s_time, gt_e_time in self.annotations[index]['timestamps']:
#             for gt_s_time, gt_e_time in timestamps_sample:
#                 s_times = torch.arange(0, num_clips).float() * duration / num_clips
#                 e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
#                 overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
#                                             e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
#                                torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips, num_clips)
#                 #overlaps (64, 64)
#                 overlaps_list.append(torch.from_numpy(overlaps))
#             overlaps_list = pad_sequence(overlaps_list, batch_first=True) #overlaps_list (k, 64, 64)
#         # Time unscaled NEED FIXED WINDOW SIZE
#         else:
#             num_clips = visual_input.shape[0]//config.DATASET.TARGET_STRIDE
#             raise NotImplementedError
#         if self.split == 'train':
#             item = {
#                 'visual_input': visual_input,
#                 'vis_mask': visual_mask,
#                 'anno_idx': index,
#                 'word_vectors': word_vectors_list, # new for dense
#                 'txt_mask': txt_mask_list, # new for dense
#                 # 'sentence_mask': torch.ones(len(self.annotations[index]['sentences']), 1), # sentence_mask (k,1) # new for dense
#                 'sentence_mask': torch.ones(len(idx_sample), 1), # sentence_mask (k,1) # new for dense
#                 'duration': duration,
#                 'map_gt': overlaps_list, # new for dense
#                 'weights_list': weights_list,
#                 'ids_list' : word_id_list
#             }
#         else:
#             item = {
#                 'visual_input': visual_input,
#                 'vis_mask': visual_mask,
#                 'anno_idx': index,
#                 'word_vectors': word_vectors_list, # new for dense
#                 'txt_mask': txt_mask_list, # new for dense
#                 # 'sentence_mask': torch.ones(len(self.annotations[index]['sentences']), 1), # sentence_mask (k,1) # new for dense
#                 'sentence_mask': torch.ones(len(idx_sample), 1), # sentence_mask (k,1) # new for dense
#                 'duration': duration,
#                 'map_gt': overlaps_list, # new for dense
#                 'weights_list': weights_list,
#                 'ids_list': word_id_list
#             }
#         return item
#
#     def __len__(self):
#         return len(self.annotations)
#
#     def get_video_features(self, vid):
#         assert config.DATASET.VIS_INPUT_TYPE == 'c3d'
#         with h5py.File(os.path.join(self.data_dir, 'sub_activitynet_v1-3.c3d.hdf5'), 'r') as f:
#             features = torch.from_numpy(f[vid]['c3d_features'][:])
#         if config.DATASET.NORMALIZE:
#             features = F.normalize(features,dim=1)
#         vis_mask = torch.ones((features.shape[0], 1))
#         return features, vis_mask

""" Dataset loader for the ActivityNet Captions dataset """
import os
import json

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from . import average_to_fixed_length
from lib.core.eval import iou
from lib.core.config import config
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import random
from IPython import embed
import nltk
import pickle

with open('/home/l/data_2/wmz/semantic_completion_network-master/data/activitynet/vocab.pkl', 'rb') as fp:
    vocabs = pickle.load(fp)

class DenseActivityNet(data.Dataset):

    # vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"](cache='../../data/glove')
    # vocab.itos.extend(['<unk>'])
    # vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    # vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    # word_embedding = nn.Embedding.from_pretrained(vocab.vectors)



    def __init__(self, split):
        super(DenseActivityNet, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split
        self.vocabs = vocabs

        self.keep_vocab = dict()
        indexs = 0
        for w, _ in vocabs['counter'].most_common(8000):
            self.keep_vocab[w] = indexs
            indexs += 1


        with open(os.path.join(self.data_dir, '{}.json'.format(split)),'r') as f:
                annotations = json.load(f)
        anno_pairs = []
        for vid, video_anno in annotations.items():
                duration = video_anno['duration']
                flag = True
                for timestamp in video_anno['timestamps']:
                    if timestamp[0] >= timestamp[1]:
                        flag = False
                if not flag:
                    continue
                anno_pairs.append(
                    {
                        'video': vid,
                        'duration': duration,
                        'sentences': video_anno['sentences'],
                        'timestamps': video_anno['timestamps']
                    }
                )
        self.annotations = anno_pairs

    def vocab_size(self):
        return len(self.keep_vocab) + 1

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        ##################################################################################################################33333333
        # 从annotations中获取视频ID和时长
        video_id = self.annotations[index]['video']
        duration = self.annotations[index]['duration']

        # 获取与视频相关的句子总数
        tot_sentence = len(self.annotations[index]['sentences'])

        # 使用所有句子的索引
        idx_sample = list(range(tot_sentence))
        idx_sample.sort()  # 这一步实际上是多余的，因为range已经是排序的

        # 获取所有句子及其时间戳
        sentence_sample = [self.annotations[index]['sentences'][idx] for idx in idx_sample]
        timestamps_sample = [self.annotations[index]['timestamps'][idx] for idx in idx_sample]
        # video_id = self.annotations[index]['video']
        # duration = self.annotations[index]['duration']
        # tot_sentence = len(self.annotations[index]['sentences'])
        #
        # P = min(9, tot_sentence + 1)
        # num_sentence = np.random.randint(1, P)
        # if num_sentence > tot_sentence:
        #     num_sentence = tot_sentence
        # # id_sentence = np.random.choice(tot_sentence, num_sentence)
        # idx_sample = random.sample(range(tot_sentence), num_sentence)
        # idx_sample.sort()
        # if self.split == 'train':
        #     sentence_sample = [self.annotations[index]['sentences'][idx] for idx in idx_sample]
        #     timestamps_sample = [self.annotations[index]['timestamps'][idx] for idx in idx_sample]
        # else:
        #     sentence_sample = self.annotations[index]['sentences']
        #     timestamps_sample = self.annotations[index]['timestamps']
        #####################################################################################################################################3333
        word_vectors_list = []
        txt_mask_list = []
        weights_list = []
        # for sentence in self.annotations[index]['sentences']:
        # print(sentence_sample)

        word_id_list= []


        for sentence in sentence_sample:
            words = []
            sentence_weights = []
            for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)):
                word = word.lower()
                if word in self.keep_vocab:
                    if 'NN' in tag:
                        sentence_weights.append(2)
                    elif 'VB' in tag:
                        sentence_weights.append(2)
                    elif 'JJ' in tag or 'RB' in tag:
                        sentence_weights.append(2)
                    else:
                        sentence_weights.append(1)
                    words.append(word)
            weights_list.append(sentence_weights)
            words_id = [self.keep_vocab[w] for w in words]

            words_id =  torch.tensor(words_id)

            word_idxs = words_id
            word_id_list.append(word_idxs)
            words_feat = [self.vocabs['id2vec'][self.vocabs['w2id'][words[0]]].astype(np.float32)]
            words_feat.extend([self.vocabs['id2vec'][self.vocabs['w2id'][w]].astype(np.float32) for w in words])

            # 正确的处理方式
            words_feat_np = np.array(words_feat)  # 先将列表转换为单一的numpy.ndarray
            words_feat_tensor = torch.tensor(words_feat_np)  # 然后将numpy.ndarray转换为PyTorch张量
            word_vectors = words_feat_tensor # word_vectors (seq, 300)
            word_vectors_list.append(word_vectors)
            txt_mask_list.append(torch.ones(word_vectors.shape[0], 1))
            # 处理每个句子的权重

        word_vectors_list = pad_sequence(word_vectors_list, batch_first=True) # word_vectors_list (k, seq, 300)
        txt_mask_list = pad_sequence(txt_mask_list, batch_first=True) # txt_mask_list (k, seq, 1)

        word_id_list = [subl.clone().detach().unsqueeze(1) for subl in word_id_list]
        word_id_list = pad_sequence(word_id_list,batch_first=True)


        weights_list = [torch.tensor(sublist).unsqueeze(1) for sublist in weights_list]
        weights_list = pad_sequence(weights_list, batch_first=True)
        visual_input, visual_mask = self.get_video_features(video_id)

        # Time scaled to same size
        if config.DATASET.NUM_SAMPLE_CLIPS > 0:
            visual_input = average_to_fixed_length(visual_input)
            num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE

            overlaps_list = []
            # for gt_s_time, gt_e_time in self.annotations[index]['timestamps']:
            for gt_s_time, gt_e_time in timestamps_sample:
                s_times = torch.arange(0, num_clips).float() * duration / num_clips
                e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
                overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
                                            e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                               torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips, num_clips)
                #overlaps (64, 64)
                overlaps_list.append(torch.from_numpy(overlaps))
            overlaps_list = pad_sequence(overlaps_list, batch_first=True) #overlaps_list (k, 64, 64)
        # Time unscaled NEED FIXED WINDOW SIZE
        else:
            num_clips = visual_input.shape[0]//config.DATASET.TARGET_STRIDE
            raise NotImplementedError
        if self.split == 'train':
            item = {
                'visual_input': visual_input,
                'vis_mask': visual_mask,
                'anno_idx': index,
                'video_idx': video_id,
                'word_vectors': word_vectors_list, # new for dense
                'txt_mask': txt_mask_list, # new for dense
                # 'sentence_mask': torch.ones(len(self.annotations[index]['sentences']), 1), # sentence_mask (k,1) # new for dense
                'sentence_mask': torch.ones(len(idx_sample), 1), # sentence_mask (k,1) # new for dense
                'duration': duration,
                'map_gt': overlaps_list, # new for dense
                'weights_list': weights_list,
                'ids_list' : word_id_list
            }
        else:
            item = {
                'visual_input': visual_input,
                'vis_mask': visual_mask,
                'anno_idx': index,
                'video_idx': video_id,
                'word_vectors': word_vectors_list, # new for dense
                'txt_mask': txt_mask_list, # new for dense
                # 'sentence_mask': torch.ones(len(self.annotations[index]['sentences']), 1), # sentence_mask (k,1) # new for dense
                'sentence_mask': torch.ones(len(idx_sample), 1), # sentence_mask (k,1) # new for dense
                'duration': duration,
                'map_gt': overlaps_list, # new for dense
                'weights_list': weights_list,
                'ids_list': word_id_list
            }
        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        assert config.DATASET.VIS_INPUT_TYPE == 'c3d'
        with h5py.File(os.path.join(self.data_dir, 'sub_activitynet_v1-3.c3d.hdf5'), 'r') as f:
            features = torch.from_numpy(f[vid]['c3d_features'][:])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask