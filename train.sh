python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_psnr.py --opt options/tisr/train_swinir_sr_tisr.json  --dist True
