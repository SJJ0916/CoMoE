##testing
#CUDA_VISIBLE_DEVICES=0 \
#python test.py \
#--config_file 'logs/RSTPReid/sdm+itc+aux_ne4_top2_layer4_head8_moe1.0_tlr1.0_aux1.0_lr3e-06_cnum9/configs.yaml'

#Trainging
CUDA_VISIBLE_DEVICES=3 \
python train_ns.py \
--name php \
--img_aug \
--batch_size 64 \
--loss_names 'sdm+itc+aux' \
--dataset_name 'ICFG-PEDES' \
--root_dir "/media/jqzhu/哈斯提·基拉/UniMoESE/data" \
--num_epoch 60 \
--lr 3e-6 \
--cnum 9 \
--num_experts 4 \
--topk 2 \
--reduction 8 \
--moe_layers 4 \
--moe_heads 8 \
--aux_factor 1.0
