#!/bin/bash

wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip -P data

unzip data/data_depth_selection.zip -d data

rm data/data_depth_selection.zip

