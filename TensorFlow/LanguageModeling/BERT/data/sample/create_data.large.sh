# BERT_LARGE="uncased_L-24_H-1024_A-16"
# BERT_LARGE_DIR=/workspace/bert/data/pretrained_models_google/$BERT_LARGE
# BERT_SRC_DIR=/workspace/bert
# python $BERT_SRC_DIR/create_pretraining_data.py \
#   --input_file=$BERT_SRC_DIR/sample_text.txt \
#   --output_file=/workspace/bert/data/sample/tf_examples.$BERT_LARGE.tfrecord \
#   --vocab_file=$BERT_LARGE_DIR/vocab.txt \
#   --do_lower_case=True \
#   --max_seq_length=512 \
#   --max_predictions_per_seq=20 \
#   --masked_lm_prob=0.15 \
#   --random_seed=12345 \
#   --dupe_factor=5

USE_BERT_LARGE=true
MAX_SEQUENCE_LENGTH=512
MAX_PREDICTIONS_PER_SEQUENCE=80
MASKED_LM_PROB=0.15
SEED=12345
DUPE_FACTOR=5
DO_LOWER_CASE="True"
N_LINES_PER_SHARD_APPROX=396000

BERT_LARGE="uncased_L-24_H-1024_A-16"
BERT_LARGE_DIR=/workspace/bert/data/pretrained_models_google/$BERT_LARGE
BERT_SRC_DIR=/workspace/bert
INPUT_FILE=$BERT_SRC_DIR/sample_text.txt
OUTPUT_DIR="/workspace/bert/data/sample/sharded"
OUTPUT_FILE=$OUTPUT_DIR/tf_examples.$BERT_LARGE.tfrecord
VOCAB_FILE="${BERT_LARGE_DIR}/vocab.txt"

if [ ! -d "$OUTPUT_DIR" ] ; then
   mkdir $OUTPUT_DIR
fi
python /workspace/bert/create_pretraining_data.py \
  --input_file=${INPUT_FILE} \
  --output_file=${OUTPUT_FILE} \
  --vocab_file=${VOCAB_FILE} \
  --do_lower_case=${DO_LOWER_CASE} \
  --max_seq_length=${MAX_SEQUENCE_LENGTH} \
  --max_predictions_per_seq=${MAX_PREDICTIONS_PER_SEQUENCE} \
  --masked_lm_prob=${MASKED_LM_PROB} \
  --random_seed=${SEED} \
  --dupe_factor=${DUPE_FACTOR}


seq 0 128 | xargs -I {} cp $OUTPUT_FILE "${OUTPUT_FILE}000{}" 
rm $OUTPUT_FILE 
