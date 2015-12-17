#!/usr/bin/env bash


if [ -z "$1" ] 
then
    exit "Missing argument"
fi

test="$1"
shift

/opt/spark/bin/spark-submit \
    --master spark://node04-ib0:7077 \
    --class "com.intel.daal.spark.rdd.tests.$test" \
    --jars /opt/intel/daal/lib/daal.jar,./spark-daal-1.0.jar \
    --driver-library-path $LD_LIBRARY_PATH  \
    --conf "spark.executorEnv.LD_LIBRARY_PATH=$LD_LIBRARY_PATH" ./spark-daal-1.0-tests.jar  "$@"
    



