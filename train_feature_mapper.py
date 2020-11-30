# Steps for offline training:
# 1. load benign pcap file
# 2. extract features
# 3. train feature mapper model and save model
import numpy as np
from kitsune.FeatureExtractor import FE

if __name__ == "__main__":
    # load benign pcap file
    packet_file = "capEC2AMAZ-O4EL3NG-172.31.69.26a.pcap"
    packet_limit = np.Inf

    max_AE = 10
    FM_grace = 5000

    # create feature extractor to get next input vector
    fe = FE(packet_file, limit=packet_limit)

    # get next input vector
    while True:
        x = fe.get_next_vector()
        if len(x) == 0:
            break
    
    # put input vector into feature mapper to train it
    # save trained mapper to file
