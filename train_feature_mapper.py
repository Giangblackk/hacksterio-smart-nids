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


if __name__ == "__main__":
    _pcap_files = ["capEC2AMAZ-O4EL3NG-172.31.69.26a.pcap.tsv"]
    _num_cluster = 10
    _limit = np.Inf
    _dataset_file = "capEC2AMAZ-O4EL3NG-172.31.69.26a.pcap.h5"
    _mapper_file = "mapper.json"
    _mapper = feature_extract_and_mapping(
        _pcap_files, _num_cluster, _limit, _dataset_file, _mapper_file
    )
    print(_mapper)

"""
if __name__ == "__main__":
    # load benign pcap file
    packet_file = "capEC2AMAZ-O4EL3NG-172.31.69.26a.pcap.tsv"
    packet_limit = np.Inf

    num_clusters = 10
    FM_grace = 10000
    AD_grace = 20000
    threshold_grace = 30000

    learning_rate = 0.1
    hidden_ratio = 0.75

    # create feature extractor to get next input vector
    fe = FE(packet_file, limit=packet_limit)

    fm = CC.corClust(fe.get_num_features())

    # get next input vector
    print("Feature Mapper training")
    curIndex = 0
    while True:
        x = fe.get_next_vector()
        if len(x) == 0:
            break

        # train feature mapper
        fm.update(x)

        curIndex += 1
        if curIndex == FM_grace:
            break

    # get trained feature mapper
    feature_map = fm.cluster2(num_clusters)

    print(feature_map)
"""
