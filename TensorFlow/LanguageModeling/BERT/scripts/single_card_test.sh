rm -rf results/* && \
    rm -rf checkpoints/* && \
    bash scripts/run_pretraining.sh 12 none 1e-4 fp32 1 10000 80000 5000 true base single
