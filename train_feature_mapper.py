# Steps for offline training:
# 1. load benign pcap file
# 2. extract features
# 3. train feature mapper model and save model
import numpy as np
from kitsune.FeatureExtractor import FE
from kitsune.KitNET import corClust as CC

if __name__ == "__main__":
    # load benign pcap file
    packet_file = "capEC2AMAZ-O4EL3NG-172.31.69.26a.pcap"
    packet_limit = np.Inf

    max_AE = 10
    FM_grace = 5000

    # create feature extractor to get next input vector
    fe = FE(packet_file, limit=packet_limit)

    fm = CC.corClust(fe.get_num_features())

    # get next input vector
    curIndex = 0
    while True:
        x = fe.get_next_vector()
        if len(x) == 0:
            break
        fm.update(x)
        print("curIndex", curIndex)
        curIndex += 1
        if curIndex == FM_grace:
            break
    
    fm.cluster(max_AE)

    # put input vector into feature mapper to train it
    # save trained mapper to file
