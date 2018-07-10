#!/bin/bash
echo '==> Filter accuracy of each epoch:'$1;
cat $1|grep accuracy|grep -v Batch;
echo '==> ...Done' 
