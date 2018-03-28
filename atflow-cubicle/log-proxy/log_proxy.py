#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2018/02/26 @Northrend
#
# export service logs using atflow log proxy
# demo codes by cocodee @github
#


from __future__ import print_function
import ConfigParser
from ava_auth import AuthFactory
import requests
import re
import uuid
from datetime import datetime
import json
import docopt
import pprint


# REMOTE_API = "http://atnet-apiserver.ava.k8s-xs.qiniu.io/"
REMOTE_API = "http://atnet-apiserver.ava-staging.k8s-xs.qiniu.io/"


def _init_():
    """
    Export service logs using atflow log proxy

    Change log:
    2018/03/28      v1.1                update remote api
    2018/02/26      v1.0                basic functions

    Usage:
        log_proxy.py                    <infile> [-s|--start -c|--check] 
                                        [--job-id=str]
        log_proxy.py                    -v|--version
        log_proxy.py                    -h|--help

    Arguments:
        <infile>                        input config file 

    Options:
        -h --help                       show this screen
        -v --version                    show script version
        -s --start                      start one log exporting job 
        -c --check                      check job status
        ------------------------------------------------------------------
        --job-id=str                    job id, needed in checking task
    """
    print('=' * 80 + '\nArguments submitted:')
    for key in sorted(args.keys()):
        print ('{:<20}= {}'.format(key.replace('--', ''), args[key]))
    print('=' * 80)


def exportlogs(ak, sk, cmd, start_time, end_time, uids=[],
               query="label[?type=='classification' && name=='pulp']",
               key="", bucket="ava-test", prefix="exportlogs/"):
    """Submit a job to export serving logs.

    Args:
        ak: Qiniu account access key.
        sk: Qiniu account secret key.
        cmd: cmd of the logs,could be pulp,facex-gender,detect...
        start_time: start time of the logs,could be formatted as 2018-02-21T03:25:00
        end_time: end time of the logs,for example, 2018-02-22T00:00:00
        uids: uid array of the logs,for example,[1380588385, 1381278422]
        query: query string of jmes path query,only the logs that match the jmes path will be exported
               jmes path website:http://jmespath.org/
        key: key of the exported file
        bucket: bucket of the exported file
        prefix: prefix of the exported file

    Returns:
        json of the job id: {"id":"5a8f83b058a9b60001ca3129"}
    """
    factory = AuthFactory(ak, sk)
    url = REMOTE_API + "v1/dataflows/exportlogs"
    if key == "":
        id = str(uuid.uuid1())
        now = datetime.now()
        nowstr = now.strftime("%Y%m%d%H%M%S")
        key = cmd + "_" + nowstr + "_" + id + ".json"
    content = {
        "cmd": cmd,
        "start_time": start_time,
        "end_time": end_time,
        "bucket": bucket,
        "prefix": prefix,
        "key": key,
        "query": query,
        # "query":"label[?type=='classification' && name=='pulp'].data|[*][?(class=='pulp')&&score>`0.9`]",
        "uid": uids
    }

    res = requests.post(url, json=content, auth=factory.get_qiniu_auth())
    ret = json.loads(res.content)
    return ret


def exportstate(ak, sk, id):
    """Get job state.

    Args:
        id: job id.

    Returns:
        json state of the job: 
        {"id":"5a8f83b058a9b60001ca3129",
        "uid":1381102889,"name":"",
        "desc":"","type":"dataflow","createTime":"2018-02-23T11:00:00.481+08:00",
        "status":"running","message":"","startTime":"2018-02-23T11:00:09.242+08:00",
        "jobType":6,"prefix":"","sourceFile":"","targetFile":"","logFile":"",
        "params":{"bucket":"ava-test","cmd":"pulp","end_time":"2018-02-22T00:00:00+08:00",
        "key":"pulp_20180223110000_a3386433-1845-11e8-85eb-f40f2431779c.json",
        "prefix":"exportlogs/","query":"label[?type=='classification' \u0026\u0026 name=='pulp']",
        "start_time":"2018-02-21T03:25:00+08:00"},
        "targetFileStatistics":{"fileCount":0,"totalImageSize":0,"totalImageDimension":0},
        "specVersion":"v1"}
    """
    factory = AuthFactory(ak, sk)
    url = REMOTE_API + "v1/dataflows/" + id
    res = requests.get(url, auth=factory.get_qiniu_auth())
    ret = json.loads(res.content)
    return ret


def main():
    conf = ConfigParser.ConfigParser()
    conf.read(args['<infile>'])
    print('Configuration file loaded')
    ak = conf.get('keys', 'ak')
    sk = conf.get('keys', 'sk')
    assert (ak and sk), 'invalid aksk'
    if args['--start']:
        cmd = conf.get('params', 'cmd')
        st = conf.get('params', 'start_time')
        et = conf.get('params', 'end_time')
        uid = [int(x) for x in conf.get('params', 'uid').split(',')] if conf.get('params', 'uid') else list()
        q = conf.get('params', 'query')
        key = conf.get('params', 'key')
        bkt = conf.get('params', 'bucket')
        pfx = conf.get('params', 'prefix')
        ret = exportlogs(ak, sk, cmd, st, et, uids=uid, query=q, key=key, bucket=bkt, prefix=pfx)
        print('=> Job id: ')
        pprint.pprint(ret)
        print('=> Job status: ')
        _ = exportstate(ak, sk, ret['id'])
        pprint.pprint(_)
        with open('job_id.tmp', 'w') as f:
            f.write(ret['id'])
        print('=> Job id temporarily saved in ./job_id.tmp')
    elif args['--check']:
        assert args['--job-id'], 'checking job status task needs one job-id'
        ret = exportstate(ak, sk, args['--job-id'])
        print('=> Job status: ')
        pprint.pprint(ret)
    else:
        print('Something weird happend ┑(￣Д ￣)┍')


if __name__ == "__main__":
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='log proxy {}'.format(version))
    _init_()
    main()
