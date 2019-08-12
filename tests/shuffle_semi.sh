for d in 01 05 1 2
do
    DIR="data/bn/REGAL-d${d}-seed1" 
    python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}--1 
    rm -r ${DIR} 
    mv ${DIR}--1 ${DIR}
done


for d in 01 05 1 2
do
    DIR="data/ppi/REGAL-d${d}-seed1" 
    python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}--1 
    rm -r ${DIR} 
    mv ${DIR}--1 ${DIR}
done

for d in 01 05 1 2
do
    DIR="data/econ/REGAL-d${d}-seed1" 
    python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}--1 
    rm -r ${DIR} 
    mv ${DIR}--1 ${DIR}
done