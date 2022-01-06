#!/bin/bash
data_path=$1

cd $data_path
declare -a subjects

for data_type in {"train","val","test"}
do
    subjects+=( $(ls ${data_type}) )
done
echo ${subjects[@]}
rm -fr /home/cackowss/data_hdd/models/CNN_benj*
rm -fr /home/cackowss/data_hdd/models/Resnet_benj_bs*

for cpt in `seq 0 9`
do
    echo $cpt
    cd ${data_path}
    mv test/sub* train/.
    mv val/sub* train/.
    cd train
    n=0
    while read line
    do
        if [ "$n" -eq "0" ]
        then
            echo "val"
            echo $line; mv $line ../val/.
        else
            echo "test"
            echo $line; mv $line ../test/.
        fi
        n=$((n+1))
    done < ../cv_group_${cpt}.txt

    cd /home/cackowss/dl_generic/
    rm -fr ../data_ssd/patches_benj/
    python3 patches_generation.py -ps 91 109 1 --contrasts rCoregToMNI_CoregEstDWI_B0_hd-BET_woNaN rCoregToMNI_CoregEstT2s_hd-BET_woNaN rCoregToMNI_CoregEstDWI_HighB_Masked_woNaN rCoregToMNI_CoregEstADC_Masked_woNaN rCoregToMNI_FLAIR_hd-BET_woNaN --data_path ../data_ssd/data_benj/ --save_path ../data_ssd/patches_benj/ -mc 15 -co Group --rescale classic -a blur gamma flip_x -ai Age NIHSS TTO DT2 HTA Trt LVO
    for i in `seq 2`
    do
        echo "Running training $i for set $cpt"
	python3 use_model.py --model Resnet -dp ../data_ssd/patches_benj/ -d 3 -ff 32 -nc 5 -e 300 -sp ../data_hdd/models/Resnet_benj_all_brain_bs16_${cpt}/ -is 91 109 -spe 200 -bs 16 --batch_norm --gpu 0 -lr 0.001 --num_classes 2 --load_ckpt --class_weights 1.39 3.57 -ai Age NIHSS TTO DT2 HTA Trt LVO >logs/Resnet_benj_all_brain_${cpt}_91_109_1_flip_blur_gamma_5c_bs16_clinical.txt
    done
#    python3 reconstruction.py -o 91 109 74 -dp ../data_hdd/models/CNN_benj_${cpt}/infered/ -ip ../data_ssd/patches_benj/patches_indices.npy -ap ../data_ssd/patches_benj/affine.npy -sp ../data_hdd/models/CNN_benj_${cpt}/ --saliency
    echo "Run $cpt done!"
done
python3 get_test_accuracy.py --model_path ../data_hdd/models/Resnet_benj_all_brain_bs16 --gt_path ../data_ssd/data_benj/classification_outputs.csv -cv > ../data_ssd/data_benj/results/Resnet_all_brain_res_state_cv_91_109_1_flip_blur_gamma_5c_bs16_clinical.txt
