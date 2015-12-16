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

import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;

import com.intel.daal.algorithms.PartialResult;
import com.intel.daal.algorithms.pca.Result;
import com.intel.daal.algorithms.pca.DistributedStep1Local;
import com.intel.daal.algorithms.pca.DistributedStep2Master;
import com.intel.daal.algorithms.pca.InputId;
import com.intel.daal.algorithms.pca.MasterInputId;
import com.intel.daal.algorithms.pca.Method;
import com.intel.daal.algorithms.pca.ResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

import scala.Tuple2;

/**
 * Wraper class for Intel DAAL's PCA distributed processing algorithms.
 *
 * @Author Zhang Zhang (zhang.zhang@intel.com)
 */
public class PCA {
	
	/**
	 * PCA computation methods:
	 * CORRELATION
	 * SVD
	 */
	public enum PCAMethod implements java.io.Serializable { 
		CORRELATION, SVD;
		
		/**
		 * Get appropriate method.
		 */
		public Method getMethod() {
			Method m = Method.correlationDense;
			switch (this) {
				case CORRELATION: m = Method.correlationDense;
				case SVD: m = Method.svdDense;
			}
			return m;
		}
	}
	
	private static Tuple2<Class<? extends Number>, PCAMethod> config;

	/**
	 * Class for PCA results
	 * scores - A nx1 NumericTable of Eigenvalues, sorted from largest
	 * 		to the smallest.
	 * loadings - A nxp NumericTable of corresponding Eigenvectors.
	 */
	public static class PCAResult {
		public NumericTable scores;
		public NumericTable loadings;
		
		PCAResult(NumericTable eigenvalues, NumericTable eigenvectors) {
			scores = eigenvalues;
			loadings = eigenvectors;
		}
	}
	
	/**
	 * Method to configure PCA computation.
	 * @param t - Floating-point data type used for intermediate computations
	 * @param m - CORRELATION or SVD method
	 */
	public static void configure(Class<? extends Number> t, PCAMethod m) {
		config = new Tuple2<Class<? extends Number>, PCAMethod>(t, m);
	}
	
	/**
	 * Compute PCA for input data.
	 * @param sc - Spark context
	 * @param dc - DAAL context
	 * @param nTables - A JavaRDD<NumericTable> contains input dataset
	 * @return A PCAResult object
	 */
	public static PCAResult compute(
			JavaSparkContext sc, DaalContext dc, JavaRDD<NumericTableWithIndex> nTables) {
		final Broadcast<Tuple2<Class<? extends Number>, PCAMethod>> configBcast = sc.broadcast(config);
    	// Local processing on all slaves 
		JavaRDD<PartialResult> partsrdd = nTables.map(
    			new Function<NumericTableWithIndex, PartialResult>() {
    				public PartialResult call(NumericTableWithIndex table) {
    					DaalContext context = new DaalContext();
    					// Create algorithm to calculate PCA decomposition using Correlation method on local nodes
    					DistributedStep1Local pcaLocal = new DistributedStep1Local(
    							context, 
    							configBcast.value()._1(),
    							configBcast.value()._2().getMethod());

    					// Set input data on local node 
    					pcaLocal.input.set(InputId.data, table.getTable(context));

    					// Compute PCA on local node 
    					PartialResult pres = pcaLocal.compute();
    					pres.pack();

    					context.dispose();
    					return pres;
    				}
    			}).cache();

    	// Finalize on master 
    	List<PartialResult> partscollection = partsrdd.collect();
        DistributedStep2Master pcaMaster = new DistributedStep2Master(dc, config._1(), config._2().getMethod());
        for (PartialResult value : partscollection) {
            value.unpack(dc);
            pcaMaster.input.add(MasterInputId.partialResults, value);
        }
        pcaMaster.compute();
        Result daalresult = pcaMaster.finalizeCompute();

        return new PCAResult(daalresult.get(ResultId.eigenValues),
        						 daalresult.get(ResultId.eigenVectors));
	}

}
