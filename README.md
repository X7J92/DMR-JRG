# Dual-task Mutual Reinforcing Embedded Joint Video Paragraph Retrieval and Grounding

we are hiring talented interns: mengzhaowangg@163.com

Video Paragraph Grounding (VPG) aims to precisely locate the most appropriate moments within a video that are relevant to a given textual paragraph query. However, existing methods typically rely on large-scale annotated temporal labels and assume that the correspondence between videos and paragraphs is known. This is impractical in real-world applications, as constructing temporal labels requires significant labor costs, and the correspondence is often unknown. To address this issue, we propose a Dual-task Mutual Reinforcing Embedded Joint Video Paragraph Retrieval and Grounding method (DMR-JRG). In this method,  retrieval and grounding tasks are mutually reinforced rather than being treated as separate issues. DMR-JRG mainly consists of two branches: a retrieval branch and a grounding branch. The retrieval branch uses inter-video contrastive learning to roughly align the global features of paragraphs and videos, reducing modality differences and constructing a coarse-grained feature space to break free from the need for correspondence between paragraphs and videos. Additionally, this coarse-grained feature space further facilitates the grounding branch in extracting fine-grained contextual representations. In the grounding branch, we achieve precise cross-modal matching and grounding by exploring the consistency between local, global, and temporal dimensions of video segments and textual paragraphs. By synergizing these dimensions, we construct a fine-grained feature space for video and textual features, greatly reducing the need for large-scale annotated temporal labels. Meanwhile, we design a grounding reinforcement retrieval module (GRRM) that brings the coarse-grained feature space of the retrieval branch closer to the fine-grained feature space of the grounding branch, thereby reinforcing retrieval branch through grounding branch, and finally achieving mutual reinforcement between tasks. Extensive experiments on three challenging datasets demonstrate the effectiveness of our proposed method.


## News
- :beers: Our paper has been submitted to the TMM.

## Framework
![alt text](imgs/pipeline.jpg)

## Main Results


#### Main results on ActivityNet Captions 
| Method | Rank1@0.3 | Rank1@0.5 | Rank1@0.7 | Rank5@0.3 | Rank5@0.5 | Rank5@0.7 |
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|
| Pool | 59.45 | 44.51 | 26.54 | 85.53 | 77.13 | 61.96 |
| Conv | 58.75 | 44.05 | 27.38 | 85.65 | 76.65 | 62.26 |

#### Main results on TACoS
| Method | Rank1@0.1 | Rank1@0.3 | Rank1@0.5 | Rank5@0.1 | Rank5@0.3 | Rank5@0.5 |
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|
| Pool | 47.59 | 37.29 | 25.32 | 70.31 | 57.81 | 45.04 |
| Conv | 46.39 | 35.17 | 25.17 | 74.46 | 56.99 | 44.24 |

## Prerequisites
- pytorch 1.1.0
- python 3.7
- torchtext
- easydict
- terminaltables


## Quick Start

Please download the visual features from [box](https://rochester.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav) or [dropbox](https://www.dropbox.com/sh/dszrtb85nua2jqe/AABGAEQhPtqBIRpGPY3gZey6a?dl=0) and save it to the `data/` folder. 


#### Training
Use the following commands for training:
```
# Evaluate "Pool" in Table 1
python moment_localization/train.py --cfg experiments/charades/2D-TAN-16x16-K5L8-pool.yaml --verbose
# Evaluate "Conv" in Table 1
python moment_localization/train.py --cfg experiments/charades/2D-TAN-16x16-K5L8-conv.yaml --verbose

# Evaluate "Pool" in Table 2
python moment_localization/train.py --cfg experiments/activitynet/2D-TAN-64x64-K9L4-pool.yaml --verbose
# Evaluate "Conv" in Table 2
python moment_localization/train.py --cfg experiments/activitynet/2D-TAN-64x64-K9L4-conv.yaml --verbose

# Evaluate "Pool" in Table 3
python moment_localization/train.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-pool.yaml --verbose
# Evaluate "Conv" in Table 3
python moment_localization/train.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-conv.yaml --verbose
```

#### Testing
Our trained model are provided in [box](https://rochester.box.com/s/5cfp7a5snvl9uky30bu7mn1cb381w91v) or [dropbox](https://www.dropbox.com/sh/27i8wvwk9cw521f/AAA4FJVDFVQZSjBoWC2x8NAIa?dl=0). Please download them to the `checkpoints` folder.

Then, run the following commands for evaluation: 
```
# Evaluate "Pool" in Table 1
python moment_localization/test.py --cfg experiments/charades/2D-TAN-16x16-K5L8-pool.yaml --verbose --split test
# Evaluate "Conv" in Table 1
python moment_localization/test.py --cfg experiments/charades/2D-TAN-16x16-K5L8-conv.yaml --verbose --split test

# Evaluate "Pool" in Table 2
python moment_localization/test.py --cfg experiments/activitynet/2D-TAN-64x64-K9L4-pool.yaml --verbose --split test
# Evaluate "Conv" in Table 2
python moment_localization/test.py --cfg experiments/activitynet/2D-TAN-64x64-K9L4-conv.yaml --verbose --split test

# Evaluate "Pool" in Table 3
python moment_localization/test.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-pool.yaml --verbose --split test
# Evaluate "Conv" in Table 3
python moment_localization/test.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-conv.yaml --verbose --split test
```
