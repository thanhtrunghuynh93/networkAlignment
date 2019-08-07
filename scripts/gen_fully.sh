# this is script used to generate fully-synthetic dataset
for NN in 1000 2000 3000 4000 5000 6000 7000 8000 9000
do 
for aver in 5 10 15 20 25 30 50
do 
    # Step1: generate a graph
    python -m generate_dataset.fully_synthetic \
    --output_path data/fully-synthetic \
    --n ${NN} \
    --aver ${aver}

    # Step2: run semi-synthetic to create target noise graph
    python -m generate_dataset.semi_synthetic \
    --input_path data/fully-synthetic/small_world-n${NN}-p${aver} \
    --d 0.01 

    # Step3: shuffle id and index of nodes in target graph
    # dictionaries will be saved at data/fully_synthetic/small_world-n${NN}-p${aver}/REGAL-d01-seed1/dictionaries/groundtruth
    DIR="data/fully-synthetic/small_world-n${NN}-p${aver}/REGAL-d01-seed1" 
    python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}--1 
    rm -r ${DIR} 
    mv ${DIR}--1 ${DIR}


    # Step4: split full dictionary into train and test files.
    python utils/split_dict.py \
    --input ${DIR}/dictionaries/groundtruth \
    --out_dir ${DIR}/dictionaries/ \
    --split 0.2


    # Step5 [optioinal]: Create features for dataset
    PS="data/fully-synthetic/small_world-n${NN}-p${aver}" 
    python -m utils.create_features \
    --input_data1 ${PS}/graphsage \
    --input_data2 ${PS}/REGAL-d01-seed1/graphsage \
    --feature_dim 300 \
    --ground_truth ${PS}/REGAL-d01-seed1/dictionaries/groundtruth
done 
done 


# After 4 steps, the source_dataset path is: data/fully-synthetic/small_world-n${NN}-p${aver}/graphsage 
# and target dataset path is: data/fully-synthetic/small_world-n${NN}-p${aver}/REGAL-d01-seed1/graphsage
# full and train, test dictionary can be found at data/fully-synthetic/small_world-n${NN}-p${aver}/REGAL-d01-seed1/dictionaries/