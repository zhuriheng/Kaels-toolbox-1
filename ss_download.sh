#!/bin/bash
echo '==> Start downloading from source station:\n'$1
url=`echo ${1%\?*}`
filename=`echo "${1##*\/}"`
domain=`echo $url|cut -d '/' -f 3`
ss='nbxs-gate-io.qiniu.com'
curl `echo $1|sed -e "s/${domain}/${ss}/g"` -o "${filename%\?*}" -H "Host: $domain"
