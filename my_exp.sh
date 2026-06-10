TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 bash ./tools/dist_train.sh /media/Storage3/wmm/ICML/Medical-OD/configs/stage1_finetune_CHAOS_5_shot.py 1 --amp
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 bash ./tools/dist_train.sh /media/Storage3/wmm/ICML/Medical-OD/configs/stage1_finetune_CHAOS_10_shot.py 1 --amp
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 bash ./tools/dist_train.sh /media/Storage3/wmm/ICML/Medical-OD/configs/stage1_finetune_CHAOS_20_shot.py 1 --amp


TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 bash ./tools/dist_train.sh /media/Storage3/wmm/ICML/Medical-OD/configs/stage1_finetune_CHAOS_MR_5_shot.py 1 --amp
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 bash ./tools/dist_train.sh /media/Storage3/wmm/ICML/Medical-OD/configs/stage1_finetune_CHAOS_MR_10_shot.py 1 --amp
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 bash ./tools/dist_train.sh /media/Storage3/wmm/ICML/Medical-OD/configs/stage1_finetune_CHAOS_MR_20_shot.py 1 --amp

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 bash ./tools/dist_train.sh /media/Storage3/wmm/ICML/Medical-OD/configs/stage1_finetune_EPFetal_5_shot.py 1 --amp
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 bash ./tools/dist_train.sh /media/Storage3/wmm/ICML/Medical-OD/configs/stage1_finetune_EPFetal_10_shot.py 1 --amp
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 bash ./tools/dist_train.sh /media/Storage3/wmm/ICML/Medical-OD/configs/stage1_finetune_EPFetal_20_shot.py 1 --amp