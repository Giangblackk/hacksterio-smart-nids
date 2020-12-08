# Steps for offline training:
# 1. load benign pcap file
# 2. extract features
# 3. train feature mapper model and save model
import numpy as np
from kitsune.FeatureExtractor import FE
from kitsune.KitNET import corClust as CC
from kitsune.netStat import netStat
import h5py
import json
import argparse


def feature_extract_and_mapping(
    pcap_file_list, num_clusters, packet_limit, out_dataset_file, out_mapper_file
):
    # get number of features
    num_feats = len(netStat().getNetStatHeaders())

    # feature mapper object
    fm = CC.corClust(num_feats)

    feat_vecs = []

    for pcap_file in pcap_file_list:
        fe = FE(pcap_file, limit=packet_limit)

        while True:
            # extract next feature vector
            x = fe.get_next_vector()
            if len(x) == 0:
                break

            fm.update(x)

            feat_vecs.append(x)

    feat_vecs = np.vstack(feat_vecs).astype(np.float32)
    print(feat_vecs.shape)
    print(feat_vecs.dtype)

    with h5py.File(out_dataset_file, "w") as f:
        f.create_dataset("dataset", data=feat_vecs)

    mapper = fm.cluster2(num_clusters)
    with open(out_mapper_file, "w") as f:
        json.dump(mapper, f)

    return mapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train feature mapper with list of pcap files"
    )
    parser.add_argument("-p", "--pcap-list", nargs="+", help="Pcap file list", required=True)
    parser.add_argument("-d", "--dataset-file", help="Path to output converted feature vector dataset")
    parser.add_argument("-n", "--num-cluster", default=5, help="Number of cluster to map features")
    parser.add_argument("-m", "--mapper-file", default="mapper.json", help="Path to output feature mapper file")
    parser.add_argument)"-l", "--limit", default=np.Inf, type=int, help="Limit number of pakcet extracted from each file")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    _args = parse_args()
    _pcap_files = _args.pcap_list
    _num_cluster = _args.num_cluster
    _limit = np.Inf
    _dataset_file = _args.dataset_file
    _mapper_file = _args.mapper_file
    _mapper = feature_extract_and_mapping(
        _pcap_files, _num_cluster, _limit, _dataset_file, _mapper_file
    )
