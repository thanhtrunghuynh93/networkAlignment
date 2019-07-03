# this is an example of creating semi-synthetic noise graph from real dataset.
# target dataset can be found at data/ppi/REGAL-d05-seed1 (source dataset is in data/ppi)

# Step1: gen target graph
python -m generate_dataset.semi_synthetic \
--input_path data/ppi \
--d 0.05 


# Step2: shuffle id and index of nodes in target graph
# dictionaries will be saved at data/ppi/REGAL-d05-seed1/dictionaries/groundtruth
DIR="data/ppi/REGAL-d05-seed1" 
python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}--1 
rm -r ${DIR} 
mv ${DIR}--1 ${DIR}

# Step3: split full dictionary into train and test files.
python utils/split_dict.py \
--input ${DIR}/dictionaries/groundtruth \
--out_dir ${DIR}/dictionaries/ \
--split 0.2

