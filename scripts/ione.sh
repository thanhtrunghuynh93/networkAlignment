
PD=data/douban
PREFIX1=online
PREFIX2=offline
TRAINRATIO=0.2

python network_alignment.py \
--source_dataset ${PD}/${PREFIX1}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
IONE \
--train_dict ${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--cuda 


