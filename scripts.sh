# run experiments
sudo python -m experiments.scapy_sniff_1

# example for freeze cpkt
CUDA_VISIBLE_DEVICES=-1 python freeze_graph.py \
    --input_meta_graph models/total2/total.ckpt.meta \
    --input_checkpoint models/total2/total.ckpt \
    --output_graph models/total2/total.pb \
    --output_node_names=ae_output_1/conv2d_1/Relu \
    --input_binary=true
