#!/bin/bash
grep 'sse3\|pni' /proc/cpuinfo > /dev/null
if [ $? -eq 0 ];  then
        echo "Supported!"
else
        echo "Not supported!"
fi
