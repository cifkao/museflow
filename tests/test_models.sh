#!/bin/bash
set -ex

dir=$(mktemp -d)
museflow model RNNGenerator --config mode_rnn_generator.yaml --logdir $dir train
museflow model RNNGenerator --config mode_rnn_generator.yaml --logdir $dir sample

dir=$(mktemp -d)
museflow model RNNSeq2Seq --config mode_transposition.yaml --logdir $dir train
museflow model RNNSeq2Seq --config mode_transposition.yaml --logdir $dir run data/modes_A3.pickle $dir/out.pickle 
