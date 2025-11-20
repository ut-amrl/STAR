#!/bin/bash

PWD=$(pwd)

mkdir -p vector_db
cd vector_db
mkdir -p volumes/milvus
curl -sfL https://gist.githubusercontent.com/TieJean/1f852ba2bf6538b80ac54a55b534cf48/raw/5d9feaca84b1f93cfa2768b5b4660ffbf6fc2899/standalone_embed.sh -o standalone_embed.sh
# bash standalone_embed.sh start

cd $PWD