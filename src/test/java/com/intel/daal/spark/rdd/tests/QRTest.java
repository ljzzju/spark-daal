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

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

import com.intel.daal.services.DaalContext;
import com.intel.daal.spark.rdd.DistributedNumericTable;
import com.intel.daal.spark.rdd.QR;

public class QRTest {

	public static void main(String[] args) {
		if (args.length < 2) {
			System.err.println("Usage: QRTest <input_file> <k>");
			System.exit(1);
		}

		String inputFile = args[0];
		int k = Integer.parseInt(args[1]);

		SparkConf conf = new SparkConf().setAppName("Test: DAAL QR on Spark");
		JavaSparkContext sc = new JavaSparkContext(conf);
		DaalContext dc = new DaalContext();
		
		JavaRDD<String> lines = sc.textFile(inputFile);
		JavaRDD<Vector> vecrdd = lines.map(
				new Function<String, Vector>() {
					public Vector call(String s) {
						String[] vals = s.split(",");
						double[] dvals = new double[vals.length];
						for (int i = 0; i < vals.length; ++i) {
							dvals[i] = Double.parseDouble(vals[i]);
						}
						return new DenseVector(dvals);
					}
				});

		DistributedNumericTable distNT = DistributedNumericTable.fromJavaVectorRDD(vecrdd, 0);
		QR.QRResult res = distNT.computeQR(sc, dc);
		TestUtils.printDistributedNumericTable(dc, "DAAL result Q: ", res.Q1, k);
	
		/* MLLib 1.5.0 crashes on this 
		RowMatrix matrix = new RowMatrix(vecrdd.rdd());
		QRDecomposition<RowMatrix, Matrix> mllibres = matrix.tallSkinnyQR(true);
		JavaRDD<Vector> mllibQ = mllibres.Q().rows().toJavaRDD();
		List<Vector> mllibQk = mllibQ.take(k);
		printResult("MLLib result: ", mllibQk);
		*/
	}

}
