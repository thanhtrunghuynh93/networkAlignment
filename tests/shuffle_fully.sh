full="../dataspace/graph/fully-synthetic/"

DIR="$full/erdos-renyi-n10000-p05/random-a1-d1"
python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

DIR="$full/erdos-renyi-n10000-p1/random-a1-d1"
python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

DIR="$full/erdos-renyi-n10000-p2/random-a1-d1"
python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

DIR="$full/erdos-renyi-n10000-p3/random-a1-d1"
python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

DIR="$full/erdos-renyi-n1000-p2/random-a1-d1"
python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

DIR="$full/erdos-renyi-n100000-p2/random-a1-d1"
python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

DIR="$full/erdos-renyi-n1000000-p2/random-a1-d1"
python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

DIR="$full/erdos-renyi-n10000000-p2/random-a1-d1"
python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}
