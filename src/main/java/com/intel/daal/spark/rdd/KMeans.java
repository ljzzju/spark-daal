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

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import com.intel.daal.algorithms.kmeans.DistributedStep1Local;
import com.intel.daal.algorithms.kmeans.DistributedStep2Master;
import com.intel.daal.algorithms.kmeans.DistributedStep2MasterInputId;
import com.intel.daal.algorithms.kmeans.InputId;
import com.intel.daal.algorithms.kmeans.Method;
import com.intel.daal.algorithms.kmeans.PartialResult;
import com.intel.daal.algorithms.kmeans.Result;
import com.intel.daal.algorithms.kmeans.ResultId;
import com.intel.daal.algorithms.kmeans.init.InitDistributedStep2Master;
import com.intel.daal.algorithms.kmeans.init.InitDistributedStep2MasterInputId;
import com.intel.daal.algorithms.kmeans.init.InitInputId;
import com.intel.daal.algorithms.kmeans.init.InitMethod;
import com.intel.daal.algorithms.kmeans.init.InitDistributedStep1Local;
import com.intel.daal.algorithms.kmeans.init.InitPartialResult;
import com.intel.daal.algorithms.kmeans.init.InitResult;
import com.intel.daal.algorithms.kmeans.init.InitResultId;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.services.DaalContext;

import scala.Tuple2;

public class KMeans {

	public enum InitializationMethod implements java.io.Serializable {
		RANDOM, DETERMINISTIC;
		
		public InitMethod getMethod() {
			InitMethod m = InitMethod.defaultDense;
			switch (this) {
				case RANDOM: m = InitMethod.randomDense;
				case DETERMINISTIC: m = InitMethod.deterministicDense;
			}
			return m;
		}
	}

	public enum ClusteringMethod implements java.io.Serializable {
		LLOYDDENSE, LLOYDCSR;
		
		public Method getMethod() {
			Method m = Method.defaultDense;
			switch (this) {
				case LLOYDCSR: m = Method.lloydCSR;
				case LLOYDDENSE: m = Method.defaultDense;
			}
			return m;
		}
	}

	public static class KMeansResult {
		public NumericTableWithIndex centroids;
		public JavaRDD<NumericTableWithIndex> assignments;
		public double goalFunc;
		
		KMeansResult(
				NumericTableWithIndex centroids, 
				JavaRDD<NumericTableWithIndex> assignments, 
				double goal) {
			this.centroids = centroids;
			this.assignments = assignments;
			this.goalFunc = goal;
		}
	}
	
	private static Tuple2<Class<? extends Number>, InitializationMethod> initConfig;
	private static Tuple2<Class<? extends Number>, ClusteringMethod> clustConfig;

	public static void initConfigure(Class<? extends Number> t, InitializationMethod m) {
		initConfig = new Tuple2<Class<? extends Number>, InitializationMethod>(t, m);
	}
	
	public static void configure(Class<? extends Number> t, ClusteringMethod m) {
		clustConfig = new Tuple2<Class<? extends Number>, ClusteringMethod>(t, m);
	}
	
	public static NumericTableWithIndex initialize(
			JavaSparkContext sc,
			DaalContext dc, 
			JavaRDD<NumericTableWithIndex> nTables,
			final long nClusters,
			final long totalNRows) {
		final Broadcast<Tuple2<Class<? extends Number>, InitializationMethod>> config = sc.broadcast(initConfig);
    	// Local processing on all slaves 
		JavaRDD<InitPartialResult> partsrdd = nTables.map(
    			new Function<NumericTableWithIndex, InitPartialResult>() {
    				public InitPartialResult call(NumericTableWithIndex table) {
    					DaalContext context = new DaalContext();
    					// Create algorithm to calculate Kmeans init 
    					InitDistributedStep1Local initLocal = new InitDistributedStep1Local(
    							context, 
    							config.value()._1(), 
    							config.value()._2().getMethod(), 
    							nClusters, 
    							totalNRows, 
    							table.getIndex());
    					// Set input data on local node 
    					initLocal.input.set(InitInputId.data, table.getTable(context));

    					// Compute on local node 
    					InitPartialResult pres = initLocal.compute();
    					pres.pack();

    					context.dispose();
    					return pres;
    				}
    			}).cache();

    	// Finalizing on master 
    	List<InitPartialResult> partscollection = partsrdd.collect();
        InitDistributedStep2Master initMaster = new InitDistributedStep2Master(
        		dc, initConfig._1(), initConfig._2().getMethod(), nClusters);
        for (InitPartialResult value : partscollection) {
            value.unpack(dc);
            initMaster.input.add(InitDistributedStep2MasterInputId.partialResults, value);
        }
        initMaster.compute();
        InitResult daalresult = initMaster.finalizeCompute();

		return new NumericTableWithIndex((long) 0, (HomogenNumericTable) daalresult.get(InitResultId.centroids));
	}
	
	static private NumericTableWithIndex doOneIteration(
			JavaSparkContext sc,
			DaalContext dc, 
			JavaRDD<NumericTableWithIndex> nTables, 
			final NumericTableWithIndex inputCentroids) {
		final Broadcast<Tuple2<Class<? extends Number>, ClusteringMethod>> config = sc.broadcast(clustConfig);
		// Local processing on all slaves 
		JavaRDD<PartialResult> partsrdd = nTables.map(
				new Function<NumericTableWithIndex, PartialResult>() {
					public PartialResult call(NumericTableWithIndex table) {
    					DaalContext context = new DaalContext();
    					// Create algorithm to calculate Kmeans 
    					DistributedStep1Local local = new DistributedStep1Local(
    							context, 
    							config.value()._1(), 
    							config.value()._2().getMethod(), 
    							inputCentroids.numOfRows());
    					// Set input data on local node 
    					local.input.set(InputId.data, table.getTable(context));
    					local.input.set(InputId.inputCentroids, inputCentroids.getTable(context));

    					// Compute on local node 
    					PartialResult pres = local.compute();
    					pres.pack();
    					context.dispose();
    					return pres;
					}
				}).cache();
		
    	// Finalizing on master 
    	List<PartialResult> partscollection = partsrdd.collect();
        DistributedStep2Master master = new DistributedStep2Master(
        		dc, clustConfig._1(), clustConfig._2().getMethod(), inputCentroids.numOfRows());
        for (PartialResult value : partscollection) {
            value.unpack(dc);
            master.input.add(DistributedStep2MasterInputId.partialResults, value);
        }
        master.compute();
        Result daalresult = master.finalizeCompute();
        
		return new NumericTableWithIndex((long) 0, (HomogenNumericTable) daalresult.get(ResultId.centroids));
	}
	
	static private KMeansResult doLastIteration(
			JavaSparkContext sc,
			DaalContext dc, 
			JavaRDD<NumericTableWithIndex> nTables, 
			final NumericTableWithIndex inputCentroids,
			final boolean assignFlag) {
		final Broadcast<Tuple2<Class<? extends Number>, ClusteringMethod>> config = sc.broadcast(clustConfig);
		// Local processing on all slaves 
		JavaPairRDD<PartialResult, NumericTableWithIndex> presrdd = nTables.mapToPair(
				new PairFunction<NumericTableWithIndex, PartialResult, NumericTableWithIndex>() {
					public Tuple2<PartialResult, NumericTableWithIndex> call(NumericTableWithIndex table) {
    					DaalContext context = new DaalContext();
    					// Create algorithm to calculate Kmeans 
    					DistributedStep1Local local = new DistributedStep1Local(
    							context, 
    							config.value()._1(), 
    							config.value()._2().getMethod(), 
    							inputCentroids.numOfRows());
    					// Set input data on local node 
    					local.input.set(InputId.data, table.getTable(context));
    					local.input.set(InputId.inputCentroids, inputCentroids.getTable(context));
    					local.parameter.setAssignFlag(assignFlag);

    					// Compute on local node 
    					PartialResult pres = local.compute();
    					pres.pack();
    					
    					if (assignFlag) {
							Result res = local.finalizeCompute();
							HomogenNumericTable tmp = (HomogenNumericTable) res.get(ResultId.assignments);
							NumericTableWithIndex assignments = new NumericTableWithIndex(table.getIndex(), tmp);

							context.dispose();
							return new Tuple2<PartialResult, NumericTableWithIndex>(pres, assignments);
    					} else {
							context.dispose();
							return new Tuple2<PartialResult, NumericTableWithIndex>(pres, null);
    					}
					}
				}).cache();
		
    	// Finalizing on master 
		JavaRDD<PartialResult> partsrdd = presrdd.keys();
		JavaRDD<NumericTableWithIndex> assignments = null; 
		if (assignFlag) {
			assignments = presrdd.values();
		}

    	List<PartialResult> partscollection = partsrdd.collect();
        DistributedStep2Master master = new DistributedStep2Master(
        		dc, clustConfig._1(), clustConfig._2().getMethod(), inputCentroids.numOfRows());
        for (PartialResult value : partscollection) {
            value.unpack(dc);
            master.input.add(DistributedStep2MasterInputId.partialResults, value);
        }
        master.compute();

        Result finalres = master.finalizeCompute();
        NumericTableWithIndex centroids = new NumericTableWithIndex((long) 0, (HomogenNumericTable) finalres.get(ResultId.centroids));
        HomogenNumericTable gf = (HomogenNumericTable) finalres.get(ResultId.goalFunction);
        double[] tmp = gf.getDoubleArray();
        return new KMeansResult(centroids, assignments, tmp[0]);
	}
	
	public static KMeansResult compute(
			JavaSparkContext sc,
			DaalContext dc, 
			JavaRDD<NumericTableWithIndex> nTables, 
			NumericTableWithIndex initCentroids,
			int iterations, 
			boolean assignFlag) {
		NumericTableWithIndex centers  = initCentroids;
		for (int i = 0; i < iterations-1; ++i) {
			centers = doOneIteration(sc, dc, nTables, centers);
		}

		return doLastIteration(sc, dc, nTables, centers, assignFlag);
	}
}
