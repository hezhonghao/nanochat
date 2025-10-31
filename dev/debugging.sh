
# some if-conditions setup (now we should got everything)

# wipe the report
python -m nanochat.report reset

# train a very small 4 layer model on the CPU
# each optimization step processes a single sequence of 1024 tokens
# we run 1000 steps of optimization (bump this to get better results)
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --total_batch_size=1024 \
    --eval_every=100\  # 250 or more; heuristic
    --eval_tokens=4096 \
    --core_metric_every=50 \
    --core_metric_max_per_task=12 \
    --sample_every=50 \
    --num_iterations=100 # we do a 1000 iters to set up a baseline
#python -m scripts.base_loss --device_batch_size=1 --split_tokens=4096
#python -m scripts.base_eval --max-per-task=16
