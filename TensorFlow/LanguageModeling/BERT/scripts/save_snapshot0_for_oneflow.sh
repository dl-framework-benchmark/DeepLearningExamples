dstpath="/home/caishenghang/model_zoo/of_model" && \
    rm -rf $dstpath && \
    python3 convert_tf_ckpt_to_of.py --tf_checkpoint_path=/results/fp32-base/ --of_dump_path="$dstpath"
