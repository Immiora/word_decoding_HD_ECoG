POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -f|--fold)
      fold="$2"
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

optuna_root=/Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/$subject/densenet_test_val_n12/31/
optuna_model=densenet_sr25_car_hfb60-300_flen0.36_lr0.0025_nout40_drop0.0197_e500.
subsets_path=/Fridge/users/julia/project_decoding_jip_janneke/data/$subject/$subject_words_with_subsets_test_val_n12.csv

cd /home/julia/Documents/project_decoding_minoes
source /home/julia/Documents/Python/anaconda3/etc/profile.d/conda.sh
conda activate torch

nice python evaluate_decoders/optuna_test_per_fold.py \
    --model_path $optuna_root$optuna_model \
    --k_fold $fold \
    --subsets_path $subsets_path \
    --save_dir $optuna_root/eval

conda deactivate
#conda activate parallel_wavegan

#echo 'Training done'
# sleep 60s
#nice python synthesize_audio/synth_speech_targets_reconstructed.py \
#    --model_path $optuna_root/eval/fold$fold/$optuna_model \
#    --checkpoint checkpoint_499 \
#    --target_set validation \
#    --synth_targets
#
#conda deactivate
#conda activate torch
#
#nice python evaluate_decoders/optuna_evaluate_per_fold.py \
#    --model_path $optuna_root/eval/fold$fold/$optuna_model \
#    --checkpoint checkpoint_999 \
#    --save_dir $optuna_root/eval/fold$fold
#
#conda deactivate