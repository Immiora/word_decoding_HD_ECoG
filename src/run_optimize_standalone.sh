#!/bin/bash
# bash run_optimize_standalone.sh \
# -t jip_janneke \
# -s subject \
# -o /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
# -m mlp \
# -e 500 \
# -n 1 \
# -j 1 \
# -r 1

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

    -r|--resume)
      resume="$2"
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

cd /home/julia/Documents/project_decoding_jip_janneke/src
source /home/julia/Documents/Python/anaconda3/etc/profile.d/conda.sh

if [ -f $save_dir/$task/$subject/$model/$task'_'$subject'_'$model'.db' ] && [ -z ${resume+x} ]; then
   echo "Database file exists but script set without resume. Use -r 1"
   exit 1
fi

tmp_dir=`mktemp -d`
res_dir=$tmp_dir/results
pic_dir=$tmp_dir/pics/decoding

echo 'Temporary results in '$tmp_dir

mkdir -p $res_dir
mkdir -p $pic_dir

if [ -n "$resume" ]; then
    mkdir -p $res_dir/$task/$subject/$model
    cp -a $save_dir/$task/$subject/$model/$task'_'$subject'_'$model'.db' $res_dir/$task/$subject/$model
    echo 'Copied model to resume'
else
    echo 'Nothing to resume'
fi

conda activate torch

for j in `seq 1 $n_jobs`; do
    echo 'job '$j
    nice python ./train_decoders/run_optuna_gpu.py \
        --task $task \
        --subject $subject \
        --save_dir $res_dir \
        --model_type $model \
        --n_trials $n_trials \
        --n_epochs $n_epochs \
        --gpu 0 \
        --load_if_exists  &> $tmp_dir/optuna_$task'_'$subject'_'$model'_'$j'.log' &
    sleep 10
done

wait
conda deactivate

mkdir -p $save_dir/$task/$subject/$model
mkdir -p ${save_dir/results/pics\/decoding}/$task/$subject/$model
cp -a $tmp_dir/*.log $res_dir/$task/$subject/$model
cp -a $res_dir/$task/$subject/$model $save_dir/$task/$subject
cp -a $pic_dir/$task/$subject/$model ${save_dir/results/pics\/decoding}/$task/$subject

trap 'rm -rf -- "$tmp_dir"' EXIT


