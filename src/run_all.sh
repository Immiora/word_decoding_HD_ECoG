#!/bin/bash
# bash run_all.sh \
# -t jip_janneke \
# -s subject \
# -o /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
# -m seq2seq \
# -e 500 \
# -n 2 \
# -j 1

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

    -e|--n_epochs)
      n_epochs="$2"
      shift # past argument
      shift # past value
      ;;

    -n|--n_trials)
      n_trials="$2"
      shift # past argument
      shift # past value
      ;;

    -j|--n_jobs)
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

# save all to tmp for speed up and copy on exit
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

# resume optimization if specified: will be conflicts with existing eval though
if [ -n "$resume" ]; then
    mkdir -p $res_dir/$task/$subject/$model
    cp -a $save_dir/$task/$subject/$model/$task'_'$subject'_'$model'.db' $res_dir/$task/$subject/$model
    echo 'Copied model to resume'
else
    echo 'Nothing to resume'
fi


cd /home/julia/Documents/project_decoding_jip_janneke/src
bash run_optimize.sh -t $task -s $subject -o $res_dir -m $model -e $n_epochs -n $n_trials -j $n_jobs
bash run_eval.sh -t $task -s $subject -o $res_dir -m $model -c $n_epochs


# copy results back to storage
mkdir -p $save_dir/$task/$subject/$model
mkdir -p ${save_dir/results/pics\/decoding}/$task/$subject/$model
cp -a $res_dir/*.log $res_dir/$task/$subject/$model
cp -a $res_dir/$task/$subject/$model $save_dir/$task/$subject
cp -a $pic_dir/$task/$subject/$model ${save_dir/results/pics\/decoding}/$task/$subject

trap 'rm -rf -- "$tmp_dir"' EXIT