export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:$(pwd)/../..

folds=(0 1 2 3 4)
data_types=('monocyte' 'neutrophil')

for data_type in ${data_types[*]}
do
    for fold in ${folds[*]}
    do
        echo "${data_type} Fold ${fold}"
        # python test.py --fold ${fold} --data_type ${data_type}
        python test_clinical_variables.py --fold ${fold} --data_type ${data_type}
    done
done