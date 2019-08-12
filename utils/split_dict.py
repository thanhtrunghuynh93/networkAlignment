from __future__ import print_function, division
import numpy as np
import os
import argparse
import pdb

def parse_args():
    """
    Example script:
    python -m utils.split_dict \
    --input /home/vinhsuhi/dataspace/graph/douban/dictionaries/groundtruth \
    --out_dir /home/vinhsuhi/dataspace/graph/douban/dictionaries \
    --split 0.2
    """

    parser = argparse.ArgumentParser(description="Split groundtruth to train and val set.")
    parser.add_argument('--input', default=None, help='Path to groundtruth')
    parser.add_argument('--out_dir', default=None, help='Path to output dir')
    parser.add_argument('--split', type=float, default=0.2, help='Train/test split')
    parser.add_argument('--seed', type=int, default=123, help='Seed of random generators')
    return parser.parse_args()

def read_dict(dict_file):
        
    all_instances = []
    if os.path.exists(dict_file):
        with open(dict_file) as fp:
            for line in fp:
                ins = line.split()
                all_instances.append([ins[0], ins[1]])
    
    return all_instances

def create_dictionary(instances, out_dir, split):
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    num_train = int(len(instances) * split)
    if num_train == 0:
        num_train += 1
    num_test = len(instances) - num_train
    np.random.shuffle(instances)

    with open("{0}/node,split={1}.train.dict".format(out_dir, split), 'wt') as f:
        for instance in instances[0:num_train]:
            f.write("%s %s\n" % (instance[0], instance[1]))
        f.close()
    print(split) 
    with open("{0}/node,split={1}.test.dict".format(out_dir, split), 'wt') as f:
        for instance in instances[num_train:num_train + num_test]:
            f.write("%s %s\n" % (instance[0], instance[1]))
        f.close()

    print("DONE!")

if __name__ == "__main__":
    args = parse_args()
    print(args)    
    np.random.seed(args.seed)
    all_instances = read_dict(args.input)
    create_dictionary(all_instances, args.out_dir, args.split)
    

