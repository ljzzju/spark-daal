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
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

import com.intel.daal.algorithms.linear_regression.Model;
import com.intel.daal.algorithms.linear_regression.prediction.PredictionMethod;
import com.intel.daal.algorithms.linear_regression.prediction.PredictionResult;
import com.intel.daal.algorithms.linear_regression.prediction.PredictionResultId;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.services.DaalContext;
import com.intel.daal.spark.rdd.DistributedNumericTable;
import com.intel.daal.spark.rdd.LinearRegression;

import scala.Tuple2;

public class LinearRegressionTest {
	
	public static void main(String[] args) {
		if (args.length < 4) {
			System.err.println(
					"Usage: LinearRegressionTest <train_input_path> <data_response_sep_position> <test_data_path> <test_label_path>");
			System.exit(1);
		}

		String trInputFile = args[0];
		String pos = args[1];
		String testDataFile = args[2];
		String testLabelFile = args[3];

		SparkConf conf = new SparkConf().setAppName("Test: DAAL LinearRegression on Spark");
		JavaSparkContext sc = new JavaSparkContext(conf);
		DaalContext dc = new DaalContext();
		
		// Load training input 
		JavaRDD<String> lines = sc.textFile(trInputFile);
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

		// Load test data 
		FileDataSource testsource = new FileDataSource(dc, testDataFile,
				DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
				DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
		testsource.loadDataBlock();
		HomogenNumericTable testData = (HomogenNumericTable) testsource.getNumericTable();
		
		// Load test responses 
		FileDataSource depsource = new FileDataSource(dc, testLabelFile,
				DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
				DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
		depsource.loadDataBlock();
		HomogenNumericTable testLabels = (HomogenNumericTable) depsource.getNumericTable();

		// Training 
		JavaPairRDD<NumericTable, NumericTable> dataWithLabels = 
				DistributedNumericTable.split(distNT, Integer.parseInt(pos));
		LinearRegression.trainingConfigure(Double.class, LinearRegression.Method.NORMEQ);
		Model LRmodel = LinearRegression.train(sc, dc, dataWithLabels);
		TestUtils.printHomogenNumericTable("DAAL Betas: ", (HomogenNumericTable) LRmodel.getBeta(), 100);
		
		// Prediction 
		LinearRegression.predictionConfigure(Double.class, PredictionMethod.defaultDense);
		PredictionResult prediction = LinearRegression.predict(dc, testData, LRmodel);
		
		HomogenNumericTable result = (HomogenNumericTable) prediction.get(PredictionResultId.prediction);
		
		TestUtils.printHomogenNumericTable("DAAL predicted: ", result, 20);
		TestUtils.printHomogenNumericTable("DAAL expected: ", testLabels, 20);
	}

}
