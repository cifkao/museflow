#!/bin/bash
set -ex

tmpdir="$(mktemp -d)"
function cleanup { rm -rf "$tmpdir"; }
trap cleanup EXIT

dir="$tmpdir/logdir"

museflow model RNNGenerator --config mode_rnn_generator.yaml --logdir "$dir" train
museflow model RNNGenerator --config mode_rnn_generator.yaml --logdir "$dir" sample --seed 42 --softmax-temperature 0.8 "$dir/out.pickle"
test -f "$dir/out.pickle"
rm -rf "$dir"

museflow model RNNSeq2Seq --config mode_transposition.yaml --logdir "$dir" train
museflow model RNNSeq2Seq --config mode_transposition.yaml --logdir "$dir" run data/modes_A3.pickle "$dir/out.pickle"
test -f "$dir/out.pickle"
rm -rf "$dir"

museflow model RNNSeq2Seq --config mode_transposition_attn.yaml --logdir "$dir" train
rm -rf "$dir"
