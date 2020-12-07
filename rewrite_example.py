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

    max_AE = 10
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
    feature_map = fm.cluster(max_AE)

    print(feature_map)

    # intialize ensemble layers and output layer
    ensembleLayers = []

    for m in feature_map:
        params = AE.dA_params(
            n_visible=len(m),
            n_hidden=0,
            lr=learning_rate,
            corruption_level=0,
            gracePeriod=0,
            hiddenRatio=hidden_ratio,
        )
        ensembleLayers.append(AE.dA(params))

    params = AE.dA_params(
        len(feature_map),
        n_hidden=0,
        lr=learning_rate,
        corruption_level=0,
        gracePeriod=0,
        hiddenRatio=hidden_ratio,
    )

    outputLayer = AE.dA(params)

    print("Anomaly Detector training")
    # put input vector into feature mapper to train it
    while True:
        x = fe.get_next_vector()
        if len(x) == 0:
            break

        # train
        S_l1 = np.zeros(len(ensembleLayers))
        for a in range(len(ensembleLayers)):
            xi = x[feature_map[a]]
            S_l1[a] = ensembleLayers[a].train(xi)

        outputLayer.train(S_l1)

        curIndex += 1
        if curIndex == AD_grace:
            break

    print("Prediction")
    # execute trained model on benign part of dataset
    RMSEs = []
    while True:
        x = fe.get_next_vector()
        if len(x) == 0:
            break

        # execute
        S_l1 = np.zeros(len(ensembleLayers))
        for a in range(len(ensembleLayers)):
            xi = x[feature_map[a]]
            S_l1[a] = ensembleLayers[a].execute(xi)
        pred = outputLayer.execute(S_l1)

        RMSEs.append(pred)

        curIndex += 1
        if curIndex == threshold_grace:
            break

    # calculate threshold
    benignSample = np.log(RMSEs)
    logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))
    print(np.min(logProbs), np.max(logProbs))
    print(np.min(RMSEs), np.max(RMSEs))

    # plot the RMSE anomaly scores
    plt.figure(figsize=(10, 5))
    fig = plt.scatter(range(len(RMSEs)), RMSEs, s=1.1, c=logProbs, cmap="RdYlGn")
    plt.yscale("log")
    plt.title("Anomaly Scores from Kitsune's Execution Phase")
    plt.ylabel("RMSE (log scaled")
    plt.xlabel("Time elapsed [min]")
    figbar = plt.colorbar()
    figbar.ax.set_ylabel("Log Probability\n ", rotation=270)
    plt.show()
    # save trained mapper to file
