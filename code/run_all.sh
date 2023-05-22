#!/bin/bash

cd $1

datasets=(airbnb clothing_review)
models=(SVM LR)
airbnb_tasks=(regression classification)
clothing_tasks=(regression multiclass classification)
text_cols_list=(text all)
label_types=(synthetic real)
conditions=(none prescaling)
params=()
modalities=all
out_file_prefix=temp
num_gen_iters=500
num_discr_iters=8


printf "Airbnb tasks\n"
for task in ${airbnb_tasks[@]}
do
    for model in ${models[@]}
    do
        if [[ $task == "classification" ]]
        then
            for label_type in ${label_types[@]}
            do
                printf "${task}"
                printf "Static pretrained embeds + Airbnb + ${model} + $task + ${label_type}\n"
                python predict.py --label-type $label_type --task $task --modalities $modalities --representation embeds --embeds-type pretrained --model $model \
                    --out-file ${out_file_prefix}_${task}_label${label_type}.csv \
                    --dataset airbnb --no-counterfactual --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                printf "Static finetuned embeds + Airbnb + ${model} + $task + $label_type\n"
                python predict.py --label-type $label_type --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}_label${label_type}.csv \
                    --dataset airbnb --no-counterfactual --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                printf "Static finetuned embeds with IPW + Airbnb + ${model} + $task + $label_type\n"
                python predict.py --label-type $label_type --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}_label${label_type}.csv \
                    --dataset airbnb --no-counterfactual --simple-ipw --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                printf "Static finetuned embeds with static GANite + Airbnb + ${model} + $task + $label_type\n"
                python predict.py --label-type $label_type --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}_label${label_type}.csv \
                    --dataset airbnb --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                printf "Static finetuned embeds with static GANite + Airbnb + ${model} + $task + $label_type + separate discriminators without t\n"
                python predict.py --label-type $label_type --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}_label${label_type}.csv \
                    --dataset airbnb --separate-discriminators --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                printf "Static finetuned embeds with static GANite + Airbnb + ${model} + $task + $label_type + separate discriminators with t\n"
                python predict.py --label-type $label_type --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}_label${label_type}.csv \
                    --dataset airbnb --separate-discriminators --discriminator-t --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters
            done
        elif [[ $task == "regression" ]]
        then
            for condition in ${conditions[@]}
            do
                if [[ $condition == "prescaling" ]]
				then
					params=( --pre-scaling )
                    printf "Regression with pre-scaling\n"
				fi
                printf "Static pretrained embeds + Airbnb + ${model} + $task\n"
                python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type pretrained --model $model \
                    --out-file ${out_file_prefix}_${task}.csv \
                    --dataset airbnb --no-counterfactual "${params[@]}" --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                printf "Static finetuned embeds + Airbnb + ${model} + $task\n"
                python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                    --dataset airbnb --no-counterfactual "${params[@]}" --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                printf "Static finetuned embeds with IPW + Airbnb + ${model} + $task\n"
                python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                    --dataset airbnb --no-counterfactual "${params[@]}" --simple-ipw --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                printf "Static finetuned embeds with static GANite + Airbnb + ${model} + $task\n"
                python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                    --dataset airbnb "${params[@]}" --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                printf "Static finetuned embeds with static GANite + Airbnb + ${model} + $task + separate discriminators without t\n"
                python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                    --dataset airbnb --separate-discriminators "${params[@]}" --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters
 
                printf "Static finetuned embeds with static GANite + Airbnb + ${model} + $task + separate discriminators with t\n"
                python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                    --dataset airbnb --separate-discriminators --discriminator-t "${params[@]}" --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters
                params=()
            done
        fi
    done
done

printf "Clothing_review tasks\n"
for task in ${clothing_tasks[@]}
do    
    for model in ${models[@]}
    do
        printf "$task\n"
        for text_cols in ${text_cols_list[@]}
        do
            printf "Using $text_cols text cols\n"
            if [[ $task == 'classification' ]]
            then
                printf "Static pretrained embeds + clothing_review + ${model} + $task + $text_cols\n"
                python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type pretrained --model $model \
                    --out-file ${out_file_prefix}_${task}.csv \
                    --dataset clothing_review --text-cols $text_cols --no-counterfactual --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                printf "Static finetuned embeds + clothing_review + ${model} + $task + $text_cols\n"
                python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                    --dataset clothing_review --text-cols $text_cols --no-counterfactual --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                printf "Static finetuned embeds with IPW + clothing_review + ${model} + $task + $text_cols\n"
                python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                    --dataset clothing_review --text-cols $text_cols --no-counterfactual --simple-ipw --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                printf "Static finetuned embeds with static GANite + clothing_review + ${model} + $task + $text_cols\n"
                python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                    --dataset clothing_review --text-cols $text_cols --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                printf "Static finetuned embeds with static GANite + clothing_review + ${model} + $task + $text_cols + separate discriminators without t\n"
                python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                    --dataset clothing_review --text-cols $text_cols --separate-discriminators --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                printf "Static finetuned embeds with static GANite + clothing_review + ${model} + $task + $text_cols + separate discriminators with t\n"
                python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                    --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                    --dataset clothing_review --text-cols $text_cols --separate-discriminators --discriminator-t --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

            elif [[ $task == "regression" ]] || [[ $task == 'multiclass' ]]
            then
                for condition in ${conditions[@]}
                do
                    if [[ $condition == "prescaling" ]]
                    then
                        params=( --pre-scaling )
                        printf "Regression with pre-scaling\n"
                    fi
                    printf "Static pretrained embeds + clothing_review + ${model} + $task + $text_cols\n"
                    python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type pretrained --model $model \
                        --out-file ${out_file_prefix}_${task}.csv \
                        --dataset clothing_review --text-cols $text_cols --no-counterfactual "${params[@]}" --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                    printf "Static finetuned embeds + clothing_review + ${model} + $task + $text_cols\n"
                    python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                        --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                        --dataset clothing_review --text-cols $text_cols --no-counterfactual "${params[@]}" --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                    printf "Static finetuned embeds with IPW + clothing_review + ${model} + $task + $text_cols\n"
                    python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                        --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                        --dataset clothing_review --text-cols $text_cols --no-counterfactual "${params[@]}" --simple-ipw --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                    printf "Static finetuned embeds with static GANite + clothing_review + ${model} + $task + $text_cols\n"
                    python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                        --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                        --dataset clothing_review --text-cols $text_cols "${params[@]}" --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                    printf "Static finetuned embeds with static GANite + clothing_review + ${model} + $task + $text_cols + separate discriminators without t\n"
                    python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                        --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                        --dataset clothing_review --text-cols $text_cols --separate-discriminators "${params[@]}" --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters

                    printf "Static finetuned embeds with static GANite + clothing_review + ${model} + $task + $text_cols + separate discriminators with t\n"
                    python predict.py --label-type real --task $task --modalities $modalities --representation embeds --embeds-type finetuned --model $model \
                        --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters --out-file ${out_file_prefix}_${task}.csv \
                        --dataset clothing_review --text-cols $text_cols --separate-discriminators --discriminator-t "${params[@]}" --num-gen-iters $num_gen_iters --num-discr-iters $num_discr_iters
                    params=()
                done
            fi
        done
    done
done