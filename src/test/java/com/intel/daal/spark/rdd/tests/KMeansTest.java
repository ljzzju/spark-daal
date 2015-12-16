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
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.spark.rdd.NumericTableWithIndex;
import com.intel.daal.spark.rdd.DistributedNumericTable;
import com.intel.daal.spark.rdd.KMeans;

public class KMeansTest {

    private static void printResult(
    		String header, 
    		DaalContext dc,
    		List<NumericTableWithIndex> assignments, 
    		double goalFunc) {
    	StringBuilder builder = new StringBuilder();
    	builder.append(header);
    	builder.append("\n");

    	int i = 0;
    	for (NumericTableWithIndex table : assignments) {
    		HomogenNumericTable nt = (HomogenNumericTable) table.getTable(dc);
    		long[] ids = nt.getLongArray();
    		for (long v : ids) {
				String tmp = String.format("%d:   %d\n", i, (int)v);
				builder.append(tmp);
				++i;
				if (i >= 10) {
					break;
				}
    		}
			builder.append("\n");
        }
        System.out.println(builder.toString());
        System.out.println(String.format("Goal function = %-6.3f", goalFunc));
    }

	/**
	 * Compute KMeans for the distributed NumericTable.
	 * @param context - DaalContext.
	 * @param nClusters - The number of clusters.
	 * @param nIterations - The number of iterations to run.
	 * @param assignFlag - Whether to get cluster assignments for all observations?
	 * @return (1) Centroids. (2) Goal function: square root of the sum of the squared distances
	 * 		of observations to their nearest center. (3) Optionally, observation assignments to
	 * 		clusters.
	 */
	public static KMeans.KMeansResult computeKMeans(
			JavaSparkContext sc,
			DaalContext context, 
			DistributedNumericTable distNT,
			int nClusters, 
			int nIterations,
			boolean assignFlag) {
		KMeans.initConfigure(Double.class, KMeans.InitializationMethod.RANDOM);
		NumericTableWithIndex initialCenters = KMeans.initialize(
				sc, context, distNT.getTables(), nClusters, distNT.numRows());
			
		KMeans.configure(Double.class, KMeans.ClusteringMethod.LLOYDDENSE);
		return KMeans.compute(sc, context, distNT.getTables(), initialCenters, nIterations, assignFlag);
	}

	public static void main(String[] args) {
		if (args.length < 3) {
			System.err.println("Usage: KMeansTest <input_file> <nClusters> <nIterations>");
			System.exit(1);
		}

		String inputFile = args[0];
		int k = Integer.parseInt(args[1]);
		int iterations = Integer.parseInt(args[2]);

		SparkConf conf = new SparkConf().setAppName("Test: DAAL KMeans on Spark");
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
		KMeans.KMeansResult res = computeKMeans(sc, dc, distNT, k, iterations, true);
		List<NumericTableWithIndex> assignments =  res.assignments.take(1);

		printResult("DAAL result: ", dc, assignments, res.goalFunc);
	}

}
