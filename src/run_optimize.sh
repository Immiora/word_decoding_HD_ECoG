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

    -e|--epochs)
      n_epochs="$2"
      shift # past argument
      shift # past value
      ;;

    -n|--trials)
      n_trials="$2"
      shift # past argument
      shift # past value
      ;;

    -j|--jobs)
      n_jobs="$2"
      shift # past argument
      shift # past value
      ;;

    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

n_trials="${n_trials:-20}"
n_jobs="${n_jobs:-5}"

source /home/julia/Documents/Python/anaconda3/etc/profile.d/conda.sh

conda activate torch

for j in `seq 1 $n_jobs`; do
    echo 'job '$j
    nice python ./train_decoders/run_optuna_gpu.py \
        --task $task \
        --subject $subject \
        --save_dir $save_dir \
        --model_type $model \
        --n_trials $n_trials \
        --n_epochs $n_epochs \
        --gpu 0 \
        --load_if_exists  &> $save_dir/optuna_$task'_'$subject'_'$model'_'$j'.log' &
    sleep 10
done

wait
conda deactivate




