#!/bin/bash

i=0;
for file in `cat "$1"|cut -f1 -d' '`;
do
  if [ ! -f "$file" ];
  then
    echo $file;
    ((i++))
  fi
done
echo 'missing file number: '${i}
