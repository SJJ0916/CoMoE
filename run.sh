##testing
#CUDA_VISIBLE_DEVICES=0 \
#python test.py \
#--config_file 'logs/ICFG-PEDES/sdm+itc+aux_lr3e-06_test/configs.yaml'



#Trainging
CUDA_VISIBLE_DEVICES=5 \
python train.py \
--name php \
--img_aug \
--batch_size 64 \
--loss_names 'sdm+itc+aux' \
--dataset_name 'RSTPReid' \
--root_dir "your data path" \
--num_epoch 60 \
--lr 3e-6 \
--cnum 9 \
--num_experts 4 \
--topk 2 \
--reduction 8 \
--moe_layers 4 \
--moe_heads 8 \
--aux_factor 1.0


#CUHK-PEDES ICFG-PEDES RSTPReid