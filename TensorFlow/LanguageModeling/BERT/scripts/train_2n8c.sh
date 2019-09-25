rm -rf results/* && \
rm -rf checkpoints/* && \
bash scripts/run_pretraining.sh 24 none 1e-4 fp16_xla 8 10000 80000 5000 true base multi
