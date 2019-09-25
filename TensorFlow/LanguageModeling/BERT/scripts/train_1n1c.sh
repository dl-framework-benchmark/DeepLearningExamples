rm -rf results/* && \
rm -rf checkpoints/* && \
export CUDA_VISIBLE_DEVICES=0 && \
bash scripts/run_pretraining.sh 4 none 1e-4 fp16_xla 1 100 300 5000 true base
