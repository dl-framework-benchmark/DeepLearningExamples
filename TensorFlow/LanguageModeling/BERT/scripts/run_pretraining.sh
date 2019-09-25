#! /bin/bash
export NCCL_DEBUG=INFO
# export CUDA_VISIBLE_DEVICES=3
echo "Container nvidia build = " $NVIDIA_BUILD_ID

SAMPLE_DIR=/workspace/bert/data/sample/sharded
WIKI_DIR=/workspace/bert/data/wikipedia_corpus/final_tfrecords_sharded
BOOKS_DIR=/workspace/bert/data/bookcorpus/final_tfrecords_sharded
BERT_CONFIG=/workspace/bert/data/pretrained_models_google/uncased_L-24_H-1024_A-16/bert_config.json
RESULTS_DIR=/results
CHECKPOINTS_DIR=/checkpoints

if [ ! -d "$WIKI_DIR" ]; then
   echo "Error! $WIKI_DIR directory missing. Please mount wikipedia dataset."
   # exit -1
else
   SOURCES="$WIKI_DIR/*"
fi
if [ ! -d "$BOOKS_DIR" ]; then
   echo "Warning! $BOOKS_DIR directory missing. Training will proceed without book corpus."
else
   SOURCES+=" $BOOKS_DIR/*"
fi
if [ ! -d "$RESULTS_DIR" ]; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
if [ ! -d "$CHECKPOINTS_DIR" ]; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be written to $RESULTS_DIR instead."
   CHECKPOINTS_DIR=$RESULTS_DIR
fi
if [ ! -f "$BERT_CONFIG" ]; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
fi

train_batch_size=${1:-14}
eval_batch_size=${2:-8}
learning_rate=${3:-"1e-4"}
precision=${4:-"fp16_xla"}
num_gpus=${5:-8}
warmup_steps=${6:-"10000"}
train_steps=${7:-1144000}
save_checkpoint_steps=${8:-5000}
create_logfile=${9:-"true"}
large_or_base=${10:-"large"}
single_or_multi_node=${11:-"single"}

PREC=""
if [ "$precision" = "fp16" ]; then
   PREC="--use_fp16"
elif [ "$precision" = "fp16_xla" ]; then
   PREC="--use_fp16 --use_xla"
elif [ "$precision" = "fp32" ]; then
   PREC=""
elif [ "$precision" = "amp" ]; then
   PREC="--amp"
elif [ "$precision" = "amp_xla" ]; then
   PREC="--amp --use_xla"
else
   echo "Unknown <precision> argument"
   exit -2
fi

if [ "$large_or_base" = "base" ]; then
   max_seq_length=128
   max_predictions_per_seq=20
   SAMPLE_DIR=/workspace/bert/data/sample/sharded-base
   BERT_CONFIG=/workspace/bert/data/pretrained_models_google/uncased_L-12_H-768_A-12/bert_config.json
elif [ "$large_or_base" = "large" ]; then
   max_seq_length=512
   max_predictions_per_seq=80
else
   echo "Unknown <large_or_base> argument"
   exit -2
fi

if [ ! -d "$SAMPLE_DIR" ]; then
   echo "Warning! $SAMPLE_DIR directory missing. Training will proceed without book corpus."
else
   SOURCES+=" $SAMPLE_DIR/*"
fi

RESULTS_DIR="$RESULTS_DIR/$precision-$large_or_base"
CHECKPOINTS_DIR="$CHECKPOINTS_DIR/$precision-$large_or_base"

echo $SOURCES
INPUT_FILES=$(eval ls $SOURCES | tr " " "\n" | awk '{printf "%s,",$1}' | sed s'/.$//')
# INPUT_FILES="/home/caishenghang/dataset/bert-test/of_wiki_seq_len_128_partial.tfrecord0000/part-r-00000"
# INIT_CHECKPOINT="results_fp32-base_4000/model.ckpt-0"
CMD="python3 /workspace/bert/run_pretraining.py"
CMD+=" --init_checkpoint=$INIT_CHECKPOINT"
CMD+=" --input_file=$INPUT_FILES"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --bert_config_file=$BERT_CONFIG"
CMD+=" --do_train=True"
CMD+=" --do_eval=False"
CMD+=" --train_batch_size=$train_batch_size"
# CMD+=" --eval_batch_size=$eval_batch_size"
CMD+=" --max_seq_length=$max_seq_length"
CMD+=" --max_predictions_per_seq=$max_predictions_per_seq"
CMD+=" --num_train_steps=$train_steps"
CMD+=" --num_warmup_steps=$warmup_steps"
CMD+=" --save_checkpoint_steps=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
# CMD+=" --report_loss"
CMD+=" $PREC"
CMD+=" --debug"
CMD+=" --dump_output_dir=$RESULTS_DIR/dump"
# export XLA_FLAGS="--xla_hlo_profile --xla_dump_to=$RESULTS_DIR/xla_hlo_profile --xla_dump_hlo_as_text"

if [ $num_gpus -gt 1 ]; then
   CMD+=" --horovod"
   if [ $single_or_multi_node = "single" ]; then
      CMD="mpiexec --allow-run-as-root -np $num_gpus --bind-to socket $CMD"
   fi
   if [ $single_or_multi_node = "multi" ]; then
      # CMD="mpiexec --allow-run-as-root -np $num_gpus -H 172.31.24.16:8,172.31.29.249:8,172.31.26.206:8,172.31.16.106:8 $CMD"
      # CMD="horovodrun -np $num_gpus -H 172.31.24.16:8,172.31.29.249:8,172.31.26.206:8,172.31.16.106:8 -p 12346 $CMD"
      # CMD="horovodrun -np $num_gpus --verbose -H oneflow-15:4,oneflow-16:4 -p 12346 $CMD"

      # if [  -n "$(uname -a | grep Ubuntu)" ]; then INTERFACE=ens3 ; else INTERFACE=eth0; fi
      # INTERFACE=ib0
      # INTERFACE=eno1
      # TCP_INTERFACE=eno1
      # TCP_INTERFACE=$INTERFACE
      MPI_PARAMS=""
      # MPI_PARAMS+=" -x NCCL_SOCKET_IFNAME=$INTERFACE "
      # MPI_PARAMS+=" -x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=mlx4_0 -x NCCL_IB_GID_INDEX=0 -x NCCL_IB_CUDA_SUPPORT=1 "
      # MPI_PARAMS+=" -x CUDA_VISIBLE_DEVICES=0,1,2,3 "
      # MPI_PARAMS+=" -x NCCL_IB_DISABLE=0 "
      # MPI_PARAMS+=" -x NCCL_DEBUG=INFO "

      MPI_PARAMS+=" -x PATH "
      MPI_PARAMS+=" -x LD_LIBRARY_PATH "

      # MPI_PARAMS+=" -mca pml ob1 "
      MPI_PARAMS+=" -mca btl ^openib "
      MPI_PARAMS+=" -mca orte_base_help_aggregate 0 "
      MPI_PARAMS+=" -mca plm_rsh_no_tree_spawn 1 "
      MPI_PARAMS+=" -mca plm_rsh_args -p2000 "
      # MPI_PARAMS+=" --mca oob_tcp_if_include $TCP_INTERFACE --mca btl_tcp_if_include $TCP_INTERFACE "
      MPI_PARAMS+=" -mca btl_tcp_if_exclude lo,docker0 "

      # MPI_PARAMS+=" --bind-to socket "
      MPI_PARAMS+=" -map-by slot "

      # MPI_PARAMS+=" -H oneflow-15:4,oneflow-16:4 "
      MPI_PARAMS+=" -H 192.168.1.15:4,192.168.1.16:4 "
      CMD="mpiexec --allow-run-as-root -np $num_gpus $MPI_PARAMS $CMD"
   fi
fi

if [ "$create_logfile" = "true" ]; then
   export GBS=$(expr $train_batch_size \* $num_gpus)
   printf -v TAG "tf_bert_1n_%s_gbs%d" "$precision" $GBS
   DATESTAMP=$(date +'%y%m%d%H%M%S')
   LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
   mkdir -p "${LOGFILE%/*}" && touch "$LOGFILE"
   printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ]; then
   $CMD
else
   (
      $CMD
   ) |& tee $LOGFILE
fi
set +x
