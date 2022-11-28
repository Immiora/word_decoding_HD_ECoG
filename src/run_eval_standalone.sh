#!/bin/bash
# bash run_eval_standalone.sh \
# -t jip_janneke \
# -s subject \
# -o /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
# -m seq2seq \
# -c 499 \
# -n 12
# -l 0

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

    -l|--trial)
      trial="$2"
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

optuna_root=$save_dir/$task/$subject/$model
echo 'optuna root: '$optuna_root

# get trial id
if [ -n "$trial" ]; then
    best_id=$trial
    echo 'using trial: '$best_id
else
    best=`grep -Po '"_number": [0-9]+,' $optuna_root/best_trial.json`
    best_id=`echo $best | grep -o -E '[0-9]+' | head -1 | sed -e 's/^\+//'`
    echo 'using best trial: '$best_id
fi

# make tmp dir
tmp_dir=`mktemp -d`
res_dir=$tmp_dir/results
pic_dir=$tmp_dir/pics/decoding

echo 'Temporary results in '$tmp_dir

mkdir -p $res_dir
mkdir -p $pic_dir
mkdir -p $res_dir/$task/$subject/$model
mkdir -p $pic_dir/$task/$subject/$model

# copy data to tmp dir

for trial in $best_id; do
#for trial in $best_id `seq 0 9`; do
    cp -a $save_dir/$task/$subject/$model/$trial $res_dir/$task/$subject/$model
    cp -a ${save_dir/results/pics\/decoding}/$task/$subject/$model/$trial $pic_dir/$task/$subject/$model
    echo 'Copied data for eval'
done

out_dir=$save_dir
save_dir=$res_dir
optuna_root=$save_dir/$task/$subject/$model

source /home/julia/Documents/Python/anaconda3/etc/profile.d/conda.sh

for trial in $best_id; do
#for trial in $best_id `seq 0 9`; do
  trial_path=$optuna_root/$trial
  model_name=`find $trial_path -maxdepth 1 -type d -name '*'$model'*' -printf "%f\n"`
  echo 'trial: '$trial
  echo $trial_path
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

# copy results back to storage
save_dir=$out_dir
cp -a $res_dir/*.log $res_dir/$task/$subject/$model
cp -au $res_dir/$task/$subject/$model $save_dir/$task/$subject
cp -au $pic_dir/$task/$subject/$model ${save_dir/results/pics\/decoding}/$task/$subject

#for i in $res_dir/$task/$subject/$model/*; do
#  cp -r -n -v "$i" $save_dir/$task/$subject/;
#done

trap 'rm -rf -- "$tmp_dir"' EXIT