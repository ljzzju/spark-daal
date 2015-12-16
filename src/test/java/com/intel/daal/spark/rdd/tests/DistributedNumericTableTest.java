/*
 *  Copyright(C) 2014-2015 Intel Corporation. All Rights Reserved.
 *
 *  The source code, information  and  material ("Material") contained herein is
 *  owned  by Intel Corporation or its suppliers or licensors, and title to such
 *  Material remains  with Intel Corporation  or its suppliers or licensors. The
 *  Material  contains proprietary information  of  Intel or  its  suppliers and
 *  licensors. The  Material is protected by worldwide copyright laws and treaty
 *  provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
 *  modified, published, uploaded, posted, transmitted, distributed or disclosed
 *  in any way  without Intel's  prior  express written  permission. No  license
 *  under  any patent, copyright  or  other intellectual property rights  in the
 *  Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
 *  implication, inducement,  estoppel or  otherwise.  Any  license  under  such
 *  intellectual  property  rights must  be express  and  approved  by  Intel in
 *  writing.
 *
 *  *Third Party trademarks are the property of their respective owners.
 *
 *  Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
 *  this  notice or  any other notice embedded  in Materials by Intel or Intel's
 *  suppliers or licensors in any way.
 *
 */
 
package com.intel.daal.spark.rdd.tests;

import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.spark.rdd.DistributedNumericTable;
import com.intel.daal.spark.rdd.NumericTableWithIndex;

import scala.Tuple2;

public class DistributedNumericTableTest {

	public static void main(String[] args) {
		if (args.length < 1) {
			System.err.println("Usage: DistributedNumericTableText <path>");
			System.exit(1);
		}

		String file1 = args[0];

		SparkConf conf = new SparkConf().setAppName("Test: DAAL Test on Spark");
		JavaSparkContext sc = new JavaSparkContext(conf);
		DaalContext dc = new DaalContext();
		
		DistributedNumericTable dnt1 = DistributedNumericTable.fromJavaVectorRDD(
				TestUtils.loadDataFromFile(sc, file1), 0);
		
        /*
		JavaRDD<NumericTableWithIndex> dnt1RDD = dnt1.getTables();
		List<NumericTableWithIndex> dnt1List = dnt1RDD.collect();
		for (NumericTableWithIndex nt : dnt1List) {
			TestUtils.printNumericTableWithIndex(dc, "NumericTable_1 with index", nt, 10);
		}
        */

		JavaPairRDD<NumericTable, NumericTable> zipped = DistributedNumericTable.split(dnt1, 10);
		List<Tuple2<NumericTable, NumericTable>> zippedList = zipped.collect();
		for (Tuple2<NumericTable, NumericTable> tup : zippedList) {
			tup._1().unpack(dc);
			tup._2().unpack(dc);
			TestUtils.printHomogenNumericTable("Table 1", (HomogenNumericTable) tup._1(), 10);
			TestUtils.printHomogenNumericTable("Table 2", (HomogenNumericTable) tup._2(), 10);
		}
	}

}
