nohup python -u train.py --is_train True --batch_size 64 --lr 0.001 --ckpt_path ckpt --backbones DPN26 --st_epoch 227 --end_epoch 400 \
	--is_fintuning True --is_cuda True --frequent 100 --data_root ./data/cifar10  --optim SGD >> ./log/DPN26-I.log
#python -u train.py --is_train True --batch_size 128 --lr 0.1 --ckpt_path ckpt --backbones mobileNetV3_large --st_epoch 0 --end_epoch 500 \
#	--is_cuda False --frequent 25 --data_root ./data/cifar10  --optim SGD





