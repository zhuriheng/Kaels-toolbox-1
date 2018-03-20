## qhash_proxy

多线程调用qhash应用抓取bucket内文件hash，目前支持`Linux`，`macOS`。

## Requirements

* 对应操作系统
* Python 2.7
* 没了

## Usage

### download exec

可以在下载完成后重命名为 qhash_proxy，并**确认文件的可执行权限**

#### v1.0 

> update 2018-03-20

* [linux](http://p4ukyt7s7.bkt.clouddn.com/log_proxy/v1.0/qhash_proxy_linux)

* [osx](http://p4ukyt7s7.bkt.clouddn.com/log_proxy/v1.0/qhash_proxy_osx)

### rock 'n roll

0. 查看使用帮助 `-h` 

1. 示例cmd：

   16线程抓取`./test.lst`里所有文件的md5和sha1（示例格式如下），输出到`./test_hash.json`，文件存储的bucket域名为`http://123456789.com/`，也就是说`http://123456789.com/test_qhash_00000000.jpg`是可访问的url。

```
$ ./qhash_proxy_linux test.lst 16 --output test_hash.json --prefix http://123456789.com/ --hash-alg md5,sha1

$ cat test.lst
test_qhash_00000000.jpg
test_qhash_00000001.jpg
test_qhash_00000002.jpg
test_qhash_00000003.jpg
test_qhash_00000004.jpg
test_qhash_00000005.jpg
test_qhash_00000006.jpg
test_qhash_00000007.jpg
```