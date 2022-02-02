#!/bin/bash
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -t|--task)
      task="$2"
      shift # past argument
      shift # past value
      ;;

    -s|--subject)
      subject="$2"
      shift # past argument
      shift # past value
      ;;

    -o|--output_dir)
      save_dir="$2"
      shift # past argument
      shift # past value
      ;;

    -m|--model)
      model="$2"
      shift # past argument
      shift # past value
      ;;

    -c|--checkpoint)
      checkpoint="$2"
      shift # past argument
      shift # past value
      ;;

    -n|--nfolds)
      nfolds="$2"
      shift # past argument
      shift # past value
      ;;

    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

eval_fold() {
  local fold=$1
  conda activate torch

  nice python evaluate_decoders/optuna_test_per_fold.py \
      --model_path $trial_path/$model_name \
      --k_fold $fold \
      --save_dir $trial_path/eval

  conda deactivate
  conda activate parallel_wavegan

  echo 'Training done'
  nice python synthesize_audio/synth_speech_targets_reconstructed.py \
      --model_path $trial_path/eval/fold$fold/$model_name \
      --checkpoint checkpoint_$checkpoint \
      --target_set validation \
      --synth_targets

  conda deactivate
  conda activate torch

  nice python evaluate_decoders/optuna_evaluate_per_fold.py \
      --model_path $trial_path/eval/fold$fold/$model_name \
      --checkpoint checkpoint_$checkpoint \
      --save_dir $trial_path/eval/fold$fold

  conda deactivate
}

set -- "${POSITIONAL[@]}" # restore positional parameters

nfolds="${nfolds:-12}"

# get best trial id
optuna_root=$save_dir/$task/$subject/$model
best=`grep -Po '"_number": [0-9]+,' $optuna_root/best_trial.json`
best_id=`echo $best | grep -o -E '[0-9]+' | head -1 | sed -e 's/^\+//'`

echo 'optuna root: '$optuna_root
echo 'best trial: '$best_id

source /home/julia/Documents/Python/anaconda3/etc/profile.d/conda.sh

for trial in $best_id `seq 0 0`; do
#for trial in $best_id `seq 0 9`; do
#for trial in $best_id; do
  trial_path=$optuna_root/$trial
  model_name=`find $trial_path -maxdepth 1 -type d -name '*'$model'*' -printf "%f\n"`
  echo 'trial: '$trial
  echo 'model_path: ' $trial_path/$model_name

  N=4 # running on all 12 folds cuda runs out of memory
  for fold in `seq 0 $((nfolds-1))`; do
    ((i=i%N)); ((i++==0)) && wait
    echo 'fold '$fold
    eval_fold "$fold" &> $save_dir/optuna_$task'_'$subject'_'$model'_eval_'$trial'_fold_'$fold'.log' &
  done

  wait
  nice python evaluate_decoders/optuna_evaluate_overview.py \
      --eval_path $trial_path/eval \
      --n_folds $nfolds

done