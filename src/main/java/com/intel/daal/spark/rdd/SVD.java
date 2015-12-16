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
 
package com.intel.daal.spark.rdd;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.intel.daal.algorithms.svd.DistributedStep3Local;
import com.intel.daal.algorithms.svd.DistributedStep3LocalInputId;
import com.intel.daal.algorithms.svd.DistributedPartialResultCollectionId;
import com.intel.daal.algorithms.svd.DistributedStep1Local;
import com.intel.daal.algorithms.svd.DistributedStep1LocalPartialResult;
import com.intel.daal.algorithms.svd.DistributedStep2Master;
import com.intel.daal.algorithms.svd.DistributedStep2MasterInputId;
import com.intel.daal.algorithms.svd.DistributedStep2MasterPartialResult;
import com.intel.daal.algorithms.svd.InputId;
import com.intel.daal.algorithms.svd.Method;
import com.intel.daal.algorithms.svd.PartialResultId;
import com.intel.daal.algorithms.svd.ResultId;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/** Intel DAAL SVD algorithm for Spark.
 * @author zzhan68
 *
 */
public class SVD {
	
	/**
	 * Computation method. Only DEFAULT is supported in this version.
	 *
	 */
	public enum SVDMethod implements java.io.Serializable { 
		DEFAULT;
		
		/**
		 * Get appropriate method.
		 */
		public Method getMethod() {
			Method m = Method.defaultDense;
			switch (this) {
				case DEFAULT: m = Method.defaultDense;
			}
			return m;
		}
	}
	
	private static Tuple2<Class<? extends Number>, SVDMethod> config;

	/**
	 * Class for SVD result.
	 * sigma - Singular values, sorted from largest to the smallest
	 * V - Right singular vectors, on the master side
	 * U - Left singular vectors (optional result), distributed across slaves
	 */
	public static class SVDResult {
		public NumericTable sigma;
		public NumericTable V;
		public DistributedNumericTable U;
		
		SVDResult(NumericTable s, NumericTable vmatrix, 
				JavaRDD<NumericTableWithIndex> umatrix, long nURows, long nUCols) {
			sigma = s; V = vmatrix; 
			U = new DistributedNumericTable(umatrix, nURows, nUCols);
		}
	}

	/**
	 * Method to configure SVD computation.
	 * @param t - The floating-point data type used for intermediate computations.
	 * @param m - DEFAULT method is the only one supported.
	 */
	static void configure(Class<? extends Number> t, SVDMethod m) {
		config = new Tuple2<Class<? extends Number>, SVDMethod>(t, m) ;
	}

	/**
	 * This method computes a SVD decomposition for input dataset.
	 * @param sc - Spark context
	 * @param dc - DAAL context
	 * @param nTables - The input dataset as a distributed NumericTable
	 * @param nRows - Total number of rows in the input
	 * @param nCols - Number of columns in the input
	 * @param computeU - Whether to compute the left singular vectors?
	 * @return A QRResult object
	 */
	static SVDResult compute(
			JavaSparkContext sc,
			DaalContext dc, 
			JavaRDD<NumericTableWithIndex> nTables, 
			long nRows, 
			long nCols, 
			boolean computeU) {

    	// Step1: Local processing on all slaves 
		// Each NumericTable in the RDD needs a index to match its local result with 
		// the local input sent back from master.
		final Broadcast<Tuple2<Class<? extends Number>, SVDMethod>> configBcast = sc.broadcast(config);
		// Each local result contains two DataCollections for step2 and step3 respectively 
		JavaPairRDD<Long, Tuple2<DataCollection, DataCollection>> partsrdd = nTables.mapToPair(
    			new PairFunction<NumericTableWithIndex, Long, Tuple2<DataCollection, DataCollection>>() {
    				public Tuple2<Long, Tuple2<DataCollection, DataCollection>> call(NumericTableWithIndex table) {
    					DaalContext context = new DaalContext();
    					// Create algorithm to calculate SVD on local nodes
    					DistributedStep1Local svdLocal = new DistributedStep1Local(
    							context, 
    							configBcast.value()._1(),
    							configBcast.value()._2().getMethod());

    					// Set input data on local node 
    					svdLocal.input.set(InputId.data, table.getTable(context));

    					// Computes on local node 
    					DistributedStep1LocalPartialResult pres = svdLocal.compute();

    					// Get input for step 2
						DataCollection step2Input = pres.get(PartialResultId.outputOfStep1ForStep2);
						step2Input.pack();
    					// Get input for step 3 (part 1 of 2)
						DataCollection step3InputLocal = pres.get(PartialResultId.outputOfStep1ForStep3);
						step3InputLocal.pack();

    					context.dispose();
    					// Pack step2Input and step3Input in a Tuple2
    					return new Tuple2<Long, Tuple2<DataCollection, DataCollection>>(
    							table.getIndex(), new Tuple2<DataCollection, DataCollection>(step2Input, step3InputLocal));
    				}
    			}).cache();
		
    	// Extract step2Input as a separate RDD
		JavaPairRDD<Long, DataCollection> step2InputRDD = partsrdd.mapToPair(
				new PairFunction<Tuple2<Long, Tuple2<DataCollection, DataCollection>>, Long, DataCollection>() {
					public Tuple2<Long, DataCollection> call(Tuple2<Long, Tuple2<DataCollection, DataCollection>> tup) {
						return new Tuple2<Long, DataCollection>(tup._1(), tup._2()._1());
					}
				});
    	
    	// Extract step3Input as a separate RDD
		JavaPairRDD<Long, DataCollection> step3InputP1RDD = partsrdd.mapToPair(
				new PairFunction<Tuple2<Long, Tuple2<DataCollection, DataCollection>>, Long, DataCollection>() {
					public Tuple2<Long, DataCollection> call(Tuple2<Long, Tuple2<DataCollection, DataCollection>> tup) {
						return new Tuple2<Long, DataCollection>(tup._1(), tup._2()._2());
					}
				});
		
		// Step2 on master 						 
		// Collect pieces of step2Input to a local ArrayList
    	List<Tuple2<Long, DataCollection>> step2MasterInputs = step2InputRDD.collect();
    	// Create algorithm for step2 on master
    	DistributedStep2Master svdMaster = new DistributedStep2Master(dc, config._1(), config._2().getMethod());
    	// Set input data for step2
    	for (Tuple2<Long, DataCollection> input : step2MasterInputs) {
    		DataCollection data = input._2();
    		data.unpack(dc);
    		svdMaster.input.add(DistributedStep2MasterInputId.inputOfStep2FromStep1, input._1().intValue(), data);
    	}
    	
    	// Computes on master 
    	DistributedStep2MasterPartialResult masterResult = svdMaster.compute();
    	// Finalize on master 
    	NumericTable s = svdMaster.finalizeCompute().get(ResultId.singularValues);
    	NumericTable v = svdMaster.finalizeCompute().get(ResultId.rightSingularMatrix);
    	
		JavaRDD<NumericTableWithIndex> uRDD = null;
    	if (computeU) {
			// Get input for step 3 (part 2 of 2)
			KeyValueDataCollection step3InputP2 = 
					masterResult.get(DistributedPartialResultCollectionId.outputOfStep2ForStep3);
			step3InputP2.pack();
			// Broadcast input for step3 (part 2 of 2) to all slaves
			final Broadcast<KeyValueDataCollection> step3InputP2Bcast = sc.broadcast(step3InputP2);

			// Step3 on slaves, only if computeU == true				 
    		uRDD = step3InputP1RDD.map(
    				new Function<Tuple2<Long, DataCollection>, NumericTableWithIndex>() {
    					public NumericTableWithIndex call(Tuple2<Long, DataCollection> tup) {
    						DaalContext context = new DaalContext();
    						
							// Retreive input for step3 (part 1 of 2)
							DataCollection qp = tup._2();
							qp.unpack(context);
    						
							// Retrieve input for step3 (part 2 of 2)
							KeyValueDataCollection step3InputP2Local = step3InputP2Bcast.value();
							step3InputP2Local.unpack(context);
							DataCollection p = (DataCollection) step3InputP2Local.get(tup._1().intValue());
    						
							// Create algorithm for step3 on local nodes
    						DistributedStep3Local svdLocal = new DistributedStep3Local(
    								context,
									configBcast.value()._1(),
									configBcast.value()._2().getMethod());
							// Set input data on local node 
    						svdLocal.input.set(DistributedStep3LocalInputId.inputOfStep3FromStep1, qp);
    						svdLocal.input.set(DistributedStep3LocalInputId.inputOfStep3FromStep2, p);
    					
							// Set input data on local node 
    						svdLocal.compute();
							// Result of this step is the left singular vectors, distributed across nodes
    						NumericTableWithIndex u = new NumericTableWithIndex(
    								tup._1(),
    								svdLocal.finalizeCompute().get(ResultId.leftSingularMatrix));
    						context.dispose();
    						return u;
    					}
    				});
    	}
    	
    	return new SVDResult(s, v, uRDD, nRows, nCols);
	}

}
