# DIR="../dataspace/graph/fully-synthetic/erdos-renyi-n1000-p25/random-a1-d1"
# python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

# DIR="../dataspace/graph/fully-synthetic/erdos-renyi-n1000-p1/random-a1-d1"
# python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

# DIR="../dataspace/graph/fully-synthetic/erdos-renyi-n5000-p5/random-a1-d1"
# python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

# DIR="../dataspace/graph/fully-synthetic/erdos-renyi-n5000-p25/random-a1-d1"
# python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

# DIR="../dataspace/graph/fully-synthetic/erdos-renyi-n5000-p1/random-a1-d1"
# python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

# DIR="../dataspace/graph/fully-synthetic/erdos-renyi-n10000-p5/random-a1-d1"
# python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

# DIR="../dataspace/graph/fully-synthetic/erdos-renyi-n10000-p25/random-a1-d1"
# python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

# DIR="../dataspace/graph/fully-synthetic/erdos-renyi-n10000-p1/random-a1-d1"
# python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}

# BASE_DIR="/home/trunght/dataspace/graph/douban/online"
# TARGET_DIR="/home/trunght/dataspace/graph/douban/offline"
# CLONE_DIR="/home/trunght/dataspace/graph/douban/"
# python -m input.dataset \
# --source_dataset $BASE_DIR/graphsage \
# --target_dataset $TARGET_DIR/graphsage \
# --groundtruth $CLONE_DIR/dictionaries/groundtruth \
# --output_dir $CLONE_DIR/statistics/

# home=$HOME
# for k in 10 20 40 60 80 100
# do
#     for p in 5
#     do
#         for seed in {100..149}
#         do
#             for ratio in 01 1
#             do
#                 BASE_DIR="/home/trunght/dataspace/graph/fully-synthetic/small-world-n1000-k$k-p$p-seed$seed"
#                 CLONE_TEMP_DIR=$BASE_DIR"/random-d$ratio/temp"        
#                 CLONE_DIR=$BASE_DIR"/random-d$ratio"        
#                 python utils/shuffle_graph.py --input_dir $CLONE_TEMP_DIR --out_dir $CLONE_DIR --seed $seed
#                 python -m input.dataset \
#                 --source_dataset $BASE_DIR/graphsage \
#                 --target_dataset $CLONE_DIR/graphsage \
#                 --groundtruth $CLONE_DIR/dictionaries/groundtruth \
#                 --output_dir $CLONE_DIR/statistics/
#             done
#         done
#     done
# done

# for k in 10
# do
#     for p in 01 05 1 2 3 4 5 6 7 8 9
#     do
#         for seed in 123
#         do
#             for ratio in 01 
#             do
#                 BASE_DIR="/home/trunght/dataspace/graph/fully-synthetic/small-world-n1000-k$k-p$p-seed$seed"
#                 CLONE_TEMP_DIR=$BASE_DIR"/random-d$ratio/temp"        
#                 CLONE_DIR=$BASE_DIR"/random-d$ratio"        
#                 python utils/shuffle_graph.py --input_dir $CLONE_TEMP_DIR --out_dir $CLONE_DIR --seed $seed
#                 python -m input.dataset \
#                 --source_dataset $BASE_DIR/graphsage \
#                 --target_dataset $CLONE_DIR/graphsage \
#                 --groundtruth $CLONE_DIR/dictionaries/groundtruth \
#                 --output_dir $CLONE_DIR/statistics/
#             done
#         done
#     done
# done

# for n in 500 1000 2000 5000 10000 20000
# do
#     for k in 10
#     do
#         for p in 5
#         do
#             for seed in 123
#             do
#                 for ratio in 01 
#                 do
#                     BASE_DIR="/home/trunght/dataspace/graph/fully-synthetic/small-world-n$n-k$k-p$p-seed$seed"
#                     CLONE_TEMP_DIR=$BASE_DIR"/random-d$ratio/temp"        
#                     CLONE_DIR=$BASE_DIR"/random-d$ratio"        
#                     python utils/shuffle_graph.py --input_dir $CLONE_TEMP_DIR --out_dir $CLONE_DIR --seed $seed
#                     python -m input.dataset \
#                     --source_dataset $BASE_DIR/graphsage \
#                     --target_dataset $CLONE_DIR/graphsage \
#                     --groundtruth $CLONE_DIR/dictionaries/groundtruth \
#                     --output_dir $CLONE_DIR/statistics/
#                 done
#             done
#         done
#     done
# done

for n in 20000
do
    for k in 10
    do
        for p in 5
        do
            for seed in 123
            do
                for ratio in 5
                do
                    BASE_DIR="/home/trunght/dataspace/graph/fully-synthetic/small-world-n$n-k$k-p$p-seed$seed"
                    CLONE_TEMP_DIR=$BASE_DIR"/random-d$ratio/temp"        
                    CLONE_DIR=$BASE_DIR"/random-d$ratio"        
                    python utils/shuffle_graph.py --input_dir $CLONE_TEMP_DIR --out_dir $CLONE_DIR --seed $seed
                    python -m input.dataset \
                    --source_dataset $BASE_DIR/graphsage \
                    --target_dataset $CLONE_DIR/graphsage \
                    --groundtruth $CLONE_DIR/dictionaries/groundtruth \
                    --output_dir $CLONE_DIR/statistics/
                done
            done
        done
    done
done
# BASE_DIR="/home/trunght/dataspace/graph/ppi/subgraphs/subgraph3"
# CLONE_DIR=$BASE_DIR"/semi-synthetic/REGAL-d005"        
# python -m input.dataset \
# --source_dataset $BASE_DIR/graphsage \
# --target_dataset $CLONE_DIR/graphsage \
# --groundtruth $CLONE_DIR/dictionaries/groundtruth \
# --output_dir $CLONE_DIR/statistics/

# BASE_DIR="/home/trunght/dataspace/graph/ppi/subgraphs/subgraph3/semi-synthetic/PALE-s9-c9-1"
# CLONE_DIR="/home/trunght/dataspace/graph/ppi/subgraphs/subgraph3/semi-synthetic/PALE-s9-c9-2"        
# python -m input.dataset \
# --source_dataset $BASE_DIR/graphsage \
# --target_dataset $CLONE_DIR/graphsage \
# --groundtruth $CLONE_DIR/dictionaries/groundtruth \
# --output_dir $CLONE_DIR/statistics/

# for k in 4
# do
#     for p in 6
#     do
#         BASE_DIR="/home/trunght/dataspace/graph/fully-synthetic/small-world-n50-k$k-p$p"
#         CLONE_TEMP_DIR=$BASE_DIR"/random-d01/temp"        
#         CLONE_DIR=$BASE_DIR"/random-d01"        
#         python utils/shuffle_graph.py --input_dir $CLONE_TEMP_DIR --out_dir $CLONE_DIR
#         python -m input.dataset \
#         --source_dataset $BASE_DIR/graphsage \
#         --target_dataset $CLONE_DIR/graphsage \
#         --groundtruth $CLONE_DIR/dictionaries/groundtruth \
#         --output_dir $CLONE_DIR/statistics/
#     done
# done