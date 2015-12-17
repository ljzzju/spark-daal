spark-daal
==========

`spark-daal` is a set of Java wrappers for using [Intel&reg; Data Analytics Acceleration
Library (Intel DAAL)](https://software.intel.com/intel-daal) on
[Spark](http://spark.apache.org). It integrates Intel DAAL's algorithms that are
capable of distributed processing into the Spark environment. It also provides
methods of converting between Spark MLlib's distributed data structures, such as
`JavaRDD<Vector>` and `RowMatrix`, and the distributed data structures readily
usable by Intel DAAL, such as `JavaRDD<NumericTable>`.

The primary purpose of this project is to serve as examples of how to use Intel
DAAL on Spark. Users can take methods implemented here and use "as-is", or they
can modify them to better fit their needs.


Requirements
============

This project depends on:

* Apache Spark Core (version 1.5.0)
* Apache Spark MLlib (version 2.10)
* Apache Commons-Lang (version 3.4)
* Intel DAAL (version 2016 and above)

To build the project, we need Apache Maven (version 3.3 and above). 

Build and Install
=================

* Edit pom.xml to set the correct path for `daal.jar` on the build system. For
   example, if DAAL installation place is "/opt/intel/daal", then the pom.xml
   should contain:
```xml
    <dependency>
  		<groupId>com.intel.daal</groupId>
  		<artifactId>daal</artifactId>
  		<version>2016</version>
  		<scope>system</scope>
  		<systemPath>/opt/intel/daal/daal.jar</systemPath>
  	</dependency>
 ```
* Build with this command:

```
 mvn clean package -DskipTests
```

Two JAR files are produced, `spark-daal-1.0.jar` (the wrapper library) and
`spark-daal-1.0-tests.jar` (tests). 

Run the Tests
=============

A simple `runtest.sh` script is provided to easily run tests included. This
script invokes `spark-submit` to submit jobs to a Spark cluster. Users will need
to edit the script to set a proper cluster master coordinate. Also, make sure
Intel DAAL runtime libraries and dependencies are included in `LD_LIBRARY_PATH`.
The best way to do this is run this command from Intel DAAL's installation
directory:

```
 %linux-prompt> source bin/daalvars.sh intel64
```

To run a test, just supply the test name, input path, and necessary arguments.
For example,

```
 ./runtest.sh PCATest <input-path> <method-name> > <number-of-principals-to-return>
```
