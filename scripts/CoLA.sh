## if using distributed learning, use the following two lines and comment out corresponding lines (Best lr may vary):
# export MULTI_GPU=0,1,2,3
# python -m torch.distributed.launch --nproc_per_node=4 --master_port 9000 main_glue.py \

export TASK_NAME=mrpc
export MULTI_GPU=4
python main_glue.py \
    --task_name $TASK_NAME \
    --multi_GPU $MULTI_GPU \
    --model_name_or_path bert-large-cased \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 150 \
    --seed 42 \
    --learning_rate 2.5e-4
