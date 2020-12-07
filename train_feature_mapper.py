# Steps for offline training:
# 1. load benign pcap file
# 2. extract features
# 3. train feature mapper model and save model
import numpy as np
from kitsune.FeatureExtractor import FE
from kitsune.KitNET import corClust as CC
from kitsune.KitNET import dA as AE
from scipy.stats import norm
from matplotlib import pyplot as plt

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
