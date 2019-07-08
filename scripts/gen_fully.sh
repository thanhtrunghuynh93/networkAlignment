# this is script used to generate fully-synthetic dataset

# Step1: generate a graph
python -m generate_dataset.fully_synthetic \
--output_path data/fully-synthetic \
--n 10000 \
--aver 5

# Step2: run semi-synthetic to create target noise graph
python -m generate_dataset.semi_synthetic \
--input_path data/fully_synthetic/erdos-renyi-n10000-p5 \
--d 0.05 

# Step3: shuffle id and index of nodes in target graph
# dictionaries will be saved at data/fully_synthetic/erdos-renyi-n10000-p5/REGAL-d05-seed1/dictionaries/groundtruth
DIR="data/fully-synthetic/erdos-renyi-n10000-p5/REGAL-d05-seed1" 
python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}--1 
rm -r ${DIR} 
mv ${DIR}--1 ${DIR}


# Step4: split full dictionary into train and test files.
python utils/split_dict.py \
--input ${DIR}/dictionaries/groundtruth \
--out_dir ${DIR}/dictionaries/ \
--split 0.2


# Step5 [optioinal]: Create features for dataset
PS="data/fully-synthetic/erdos-renyi-n10000-p5" 
python -m utils.create_features \
--input_data1 ${PS}/graphsage \
--input_data2 ${PS}/REGAL-d05-seed1/graphsage \
--feature_dim 300 \
--ground_truth ${PS}/REGAL-d05-seed1/dictionaries/groundtruth



# After 4 steps, the source_dataset path is: data/fully-synthetic/erdos-renyi-n10000-p5/graphsage 
# and target dataset path is: data/fully-synthetic/erdos-renyi-n10000-p5/REGAL-d05-seed1/graphsage
# full and train, test dictionary can be found at data/fully-synthetic/erdos-renyi-n10000-p5/REGAL-d05-seed1/dictionaries/