rm -rf results/* && \
rm -rf checkpoints/* && \
export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
bash scripts/run_pretraining.sh 24 none 1e-4 fp16_xla 4 5000 2285000 5000 true base
