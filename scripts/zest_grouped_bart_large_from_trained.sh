# 538 tasks in total

CUDA_VISIBLE_DEVICES=0 \
python cli_grouped.py \
 --learning_rate 1e-5 \
 --seed 1 \
 --dataset zest_grouped \
 --inner_bsz 16 \
 --adapter_dim 32 \
 --do_train \
 --freeze_embeds \
 --total_steps 2020 \
 --warmup_steps 120 \
 --max_grad_norm 0.1 \
 --weight_decay 0.01 \
 --model facebook/bart-large \
 --output_dir ./out/bart-large-zest-from-trained  \
 --train_file ./data/zest_train.jsonl \
 --predict_file data/zest_dev.jsonl \
 --train_batch_size 1 \
 --gradient_accumulation_steps 1 \
 --predict_batch_size 16 \
 --max_input_length 128 \
 --max_output_length 64 \
 --eval_period 1076 \
 --num_train_epochs 120;
 # --do_predict \
 # --max_input_length 32 \
  # --checkpoint ./out/bart-large-zest55-reproduce/last-model.pt \
# cd ~/zest/bin
# python evaluate-zest.py -p ~/zs/out/bart-large-zest-from-trained/predictions.txt -d ~/zest/data/dev.jsonl -o ~/zs/out/bart-large-zest-from-trained/results.json

