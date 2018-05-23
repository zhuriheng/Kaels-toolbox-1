## log_proxy

调用 atflow 的日志服务获取对应日志，目前支持`Linux`，`macOS`。

## Requirements

* 对应操作系统
* Python 2.7
* 没了

## Usage

### download exec

可以在下载完成后重命名为 log_proxy，并**确认文件的可执行权限**

#### v1.2 

> update 2018-05-23

* [linux](http://p4ukyt7s7.bkt.clouddn.com/log_proxy/v1.2/log_proxy_linux)

* [mac](http://p4ukyt7s7.bkt.clouddn.com/log_proxy/v1.2/log_proxy_mac)

#### v1.1 

> update 2018-04-02

* [linux](http://p4ukyt7s7.bkt.clouddn.com/log_proxy/v1.1/log_proxy_linux)

* [mac](http://p4ukyt7s7.bkt.clouddn.com/log_proxy/v1.1/log_proxy_mac)

#### v1.0 

> update 2018-02-28

* [linux](http://p4ukyt7s7.bkt.clouddn.com/log_proxy/v1.0/log_proxy_linux)

* [mac](http://p4ukyt7s7.bkt.clouddn.com/log_proxy/v1.0/log_proxy_mac)

### rock 'n roll

0. 查看使用帮助 `-h` 

1. 提交一个日志查询的 job：

  * 填写配置文件 `log_proxy.conf`: 

    参考[模板](./log_proxy.conf)

    目前仅支持 `avatest@qiniu.com` 账号。

    ```
    $ cat log_proxy.conf
    [keys]
    ak = xxx	# 填账号的ak
    sk = xxx	# 填账号的sk

    [params]
    cmd = pulp	# 抓取日志的目标服务 
    start_time = 2018-02-25T18:00:00	# 日志的起始时间戳
    end_time = 2018-02-27T18:00:00	# 日志的结束时间戳
    uid = [123456789,987654321]	# uid过滤列表，不限则留空 
    key = # 自定义日志文件存储的key，建议留空
    bucket = ava-test	# 自定义日志文件存储的空间，建议不修改
    prefix = exportlogs/	# 自定义日志文件存储的前缀，建议不修改
    query = label[?type=='classification' && name=='pulp']	# 指定查询条件
    ```

  * 提交 job：

      ```
      $ ./log_proxy_osx log_proxy.conf -s

      ================================================================================
      Arguments submitted:
      check               = False
      help                = False
      job-id              = None
      start               = True
      version             = False
      <infile>            = log_proxy.conf
      ================================================================================
      Configuration file loaded
      => Job id:
      {u'id': u'5a965399c19e1f000168913b'}
      => Job status:
      {u'createTime': u'2018-02-28T15:00:41.167+08:00',
       u'desc': u'',
       u'id': u'5a965399c19e1f000168913b',
       u'jobType': 6,
       u'logFile': u'',
       u'message': u'',
       u'name': u'',
       u'params': {u'bucket': u'ava-test',
                   u'cmd': u'pulp',
                   u'end_time': u'2018-02-27T18:00:00+08:00',
                   u'key': u'pulp_20180228150041_16a4f0cf-1c55-11e8-8cea-784f435ed2b3.json',
                   u'prefix': u'exportlogs/',
                   u'query': u"label[?type=='classification' && name=='pulp']",
                   u'start_time': u'2018-02-25T18:00:00+08:00',
                   u'uid': []},
       u'prefix': u'',
       u'sourceFile': u'',
       u'specVersion': u'v1',
       u'status': u'pending_start',
       u'targetFile': u'',
       u'targetFileStatistics': {u'fileCount': 0,
                                 u'totalImageDimension': 0,
                                 u'totalImageSize': 0},
       u'type': u'dataflow',
       u'uid': 1381102889}
      => Job id temporarily saved in ./job_id.tmp
      ```

2. 查询已有 job 的状态：

   最近一次提交的job id备份在`./job_id.tmp`。

   ```
   $ ./log_proxy_osx log_proxy.conf -c --job-id 5a965399c19e1f000168913b

   ================================================================================
   Arguments submitted:
   check               = True
   help                = False
   job-id              = 5a965399c19e1f000168913b
   start               = False
   version             = False
   <infile>            = log_proxy.conf
   ================================================================================
   Configuration file loaded
   => Job status:
   {u'createTime': u'2018-02-28T15:00:41.167+08:00',
    u'desc': u'',
    u'id': u'5a965399c19e1f000168913b',
    u'jobType': 6,
    u'logFile': u'',
    u'message': u'',
    u'name': u'',
    u'params': {u'bucket': u'ava-test',
                u'cmd': u'pulp',
                u'end_time': u'2018-02-27T18:00:00+08:00',
                u'key': u'pulp_20180228150041_16a4f0cf-1c55-11e8-8cea-784f435ed2b3.json',
                u'prefix': u'exportlogs/',
                u'query': u"label[?type=='classification' && name=='pulp']",
                u'start_time': u'2018-02-25T18:00:00+08:00',
                u'uid': []},
    u'prefix': u'',
    u'sourceFile': u'',
    u'specVersion': u'v1',
    u'startTime': u'2018-02-28T15:00:48.466+08:00',
    u'status': u'done',
    u'targetFile': u'',
    u'targetFileStatistics': {u'fileCount': 0,
                              u'totalImageDimension': 0,
                              u'totalImageSize': 0},
    u'type': u'dataflow',
    u'uid': 1381102889}
   ```

3. 下载日志文件：

   如果查询job状态返回`status`为done，则可以下载对应日志了。

   根据`bucket`,`prefix`,`key`3个字段可以找到日志的存储位置，具体下载方式不做赘述。


