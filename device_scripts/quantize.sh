#!/bin/sh
vai_q_tensorflow quantize --input_frozen_graph float/total.pb \
    --input_fn input_fn.calib_input \
    --output_dir quantized_1000/ \
    --input_nodes input0,input1,input2,input3,input4,input5,input6,input7,input8,input9 \
    --output_nodes ae_output_1/conv2d_1/Relu \
    --input_shapes ?,1,1,10:?,1,1,6:?,1,1,3:?,1,1,6:?,1,1,9:?,1,1,20:?,1,1,25:?,1,1,10:?,1,1,2:?,1,1,9 --calib_iter 1000
