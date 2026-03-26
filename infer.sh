
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --rdzv_endpoint=localhost:8000 \
inference_action.py \
  --num_workers 4 \
  --log_every 10 \
  --num_replicate 1 \
  --num_shard 8 \