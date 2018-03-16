#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# python -u $DIR/prepare_blued.py --dataset blued --set 0920-test --target /disk2/Northrend/blued/dataset/recordio/blued_v10/blued_0920-test.lst  --root /disk2/Northrend/blued/dataset/blued_0920/ --class-names /disk2/Northrend/blued/dataset/blued_0920/label.lst
python -u $DIR/prepare_blued.py --dataset blued --set 0920-train --target /disk2/Northrend/blued/dataset/recordio/blued_v10/blued_0920-train.lst  --root /disk2/Northrend/blued/dataset/blued_0920/ --class-names /disk2/Northrend/blued/dataset/blued_0920/label.lst
