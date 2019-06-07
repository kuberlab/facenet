#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ckpt_dir=$1
output_dir=$2

usage="$0 <ckpt_dir> <output_dir>"

if [ -z "$ckpt_dir" ]
then
  echo $usage
  exit 1
fi

if [ -z "$output_dir" ]
then
  echo $usage
  exit 1
fi

mkdir -p $output_dir
python $SCRIPT_DIR/export_checkpoint.py $ckpt_dir --output $output_dir/facenet.pb
python $SCRIPT_DIR/openvino_converter.py --training_dir $output_dir --target CPU --facenet \
 --facenet_graph $output_dir/facenet.pb

# Post actions
rm -rf $output_dir/openvino
mv $output_dir/facenet/* $output_dir/
rm -rf $output_dir/facenet