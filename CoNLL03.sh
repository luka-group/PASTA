## if using distributed learning, use the following two lines and comment out corresponding lines (Best lr may vary):
# export MULTI_GPU=0,1,2,3
# python -m torch.distributed.launch --nproc_per_node=4 --master_port 9001  main_ner.py \


export MULTI_GPU=4
python main_ner.py \
    --dataset_name conll2003 \
    --task_name ner \
    --multi_GPU $MULTI_GPU \
    --model_name_or_path bert-large-cased \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 100 \
    --seed 42 \
    --learning_rate 3e-3