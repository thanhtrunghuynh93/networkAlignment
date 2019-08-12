
for s in {1..20}
do

	python utils/random_delete_nodes.py --input $HOME/dataspace/graph/bn-fly-drosophila_medulla_1 --ratio 0.1 --seed ${s}
	python utils/random_delete_nodes.py --input $HOME/dataspace/graph/bn-fly-drosophila_medulla_1 --ratio 0.2 --seed ${s}
	python utils/random_delete_nodes.py --input $HOME/dataspace/graph/bn-fly-drosophila_medulla_1 --ratio 0.3 --seed ${s}
	python utils/random_delete_nodes.py --input $HOME/dataspace/graph/bn-fly-drosophila_medulla_1 --ratio 0.4 --seed ${s}
	python utils/random_delete_nodes.py --input $HOME/dataspace/graph/bn-fly-drosophila_medulla_1 --ratio 0.5 --seed ${s}

done





# DIR="$HOME/dataspace/graph/bn-fly-drosophila_medulla_1/del-nodes-p1"
# python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

# DIR="$HOME/dataspace/graph/bn-fly-drosophila_medulla_1/del-nodes-p2"
# python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

# DIR="$HOME/dataspace/graph/bn-fly-drosophila_medulla_1/del-nodes-p3"
# python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

# DIR="$HOME/dataspace/graph/bn-fly-drosophila_medulla_1/del-nodes-p4"
# python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

# DIR="$HOME/dataspace/graph/bn-fly-drosophila_medulla_1/del-nodes-p5"
# python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}
