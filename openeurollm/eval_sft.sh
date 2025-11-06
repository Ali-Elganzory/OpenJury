#!/bin/bash
#SBATCH --job-name=eval_sft
#SBATCH --output=slurm_logs/%j.%x.%N.out
#SBATCH --error=slurm_logs/%j.%x.%N.err
#SBATCH --time=00-02:00:00
#SBATCH --partition=accelerated-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alielganzory@hotmail.com

export WORK_DIR=/home/hk-project-p0024002/fr_ae293/work/hkfswork/fr_ae293-ra
export TMPDIR=$WORK_DIR/.tmp
export TMP=$TMPDIR

source .venv/bin/activate

EXPERIMENTS=(llama3.1_8b_sft__1760768879)
FINAL_MODEL_BASE_COMPETITORS=(VLLM/allenai/Llama-3.1-Tulu-3-8B-SFT)
OVER_TIME_BASE_COMPETITORS=(VLLM/meta-llama/Llama-3.1-8B-Instruct)
JUDGE_MODEL=VLLM/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
FINAL_MODEL_DATASETS=(alpaca-eval arena-hard m-arena-hard)
OVER_TIME_DATASETS=(alpaca-eval)

# Load env vars fron .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Evaluation function
function evaluate {
    DATASET=$1
    COMPETITOR_A=$2
    COMPETITOR_B=$3
    JUDGE_MODEL=${4:-$JUDGE_MODEL}

    python openjury/generate_and_evaluate.py \
        --dataset $DATASET \
        --model_A $COMPETITOR_A \
        --model_B $COMPETITOR_B \
        --judge_model $JUDGE_MODEL
        # --correct_order_bias
        # --n_instructions 10
}

# Final model performance
for EXP_NAME in ${EXPERIMENTS[@]}; do
    EXP_PATH=VLLM/$WORK_DIR/open-instruct/output/$EXP_NAME
    
    for BASE_COMPETITOR in ${FINAL_MODEL_BASE_COMPETITORS[@]}; do
        for DATASET in ${FINAL_MODEL_DATASETS[@]}; do
            evaluate $DATASET $BASE_COMPETITOR $EXP_PATH
        done
    done
done

# Over time performance
for EXP_NAME in ${EXPERIMENTS[@]}; do
    EXP_PATH=$WORK_DIR/open-instruct/output/$EXP_NAME

    EXP_COMPETITORS=($(ls $EXP_PATH | grep model_))
    for EXP_COMPETITOR in ${EXP_COMPETITORS[@]}; do
        EXP_COMPETITOR=VLLM/$EXP_PATH/$EXP_COMPETITOR

        for BASE_COMPETITOR in ${OVER_TIME_BASE_COMPETITORS[@]}; do
            for DATASET in ${OVER_TIME_DATASETS[@]}; do
                evaluate $DATASET $BASE_COMPETITOR $EXP_COMPETITOR
            done
        done
    done
done
