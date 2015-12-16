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
import org.apache.spark.broadcast.Broadcast;

import com.intel.daal.algorithms.linear_regression.Model;
import com.intel.daal.algorithms.linear_regression.prediction.PredictionBatch;
import com.intel.daal.algorithms.linear_regression.prediction.PredictionInputId;
import com.intel.daal.algorithms.linear_regression.prediction.PredictionMethod;
import com.intel.daal.algorithms.linear_regression.prediction.PredictionResult;
import com.intel.daal.algorithms.linear_regression.training.MasterInputId;
import com.intel.daal.algorithms.linear_regression.training.PartialResult;
import com.intel.daal.algorithms.linear_regression.training.TrainingDistributedStep1Local;
import com.intel.daal.algorithms.linear_regression.training.TrainingDistributedStep2Master;
import com.intel.daal.algorithms.linear_regression.training.TrainingInputId;
import com.intel.daal.algorithms.linear_regression.training.TrainingMethod;
import com.intel.daal.algorithms.linear_regression.training.TrainingResult;
import com.intel.daal.algorithms.linear_regression.training.TrainingResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

import scala.Tuple2;

/**
 * Class that wraps Intel DAAL Linear Regression distributed processing algorithms.
 * @author zzhan68
 *
 */
public class LinearRegression {
	
	/**
	 * Linear regression model training methods: 
	 * NORMEQ - Norm equation
	 * QR - QR methods
	 */
	public enum Method implements java.io.Serializable { 
		NORMEQ, QR;
		
		/**
		 * Get appropriate method.
		 */
		public TrainingMethod getMethod() {
			TrainingMethod m = TrainingMethod.normEqDense;
			switch (this) {
				case NORMEQ: m = TrainingMethod.normEqDense;
				case QR: m = TrainingMethod.qrDense;
			}
			return m;
		}
	}
	
	private static Tuple2<Class<? extends Number>, Method> trConfig;
	private static PredictionMethod predictMethod;
	private static Class<? extends Number> predictFpType;
	
	/**
	 * Method to configure linear regression model training.
	 * @param fpType - The floating-point data type used for intermediate computations.
	 * @param method - NORMEQ or QR method.
	 */
	public static void trainingConfigure(
			Class<? extends Number> fpType, Method method) {
		trConfig = new Tuple2<Class<? extends Number>, Method>(fpType, method);
	}

	/**
	 * Method to configure prediction.
	 * @param fpType - The floating-point data type used for intermediate computations.
	 * @param method - defaultDense is the only supported method.
	 */
	public static void predictionConfigure(
			 Class<? extends Number> fpType, PredictionMethod method) {
		predictFpType = fpType;
		predictMethod = method;
	}

	/**
	 * Train a linear regression model using known data (independent variables 
	 * and responses).
	 * @param sc - Spark context 
	 * @param dc - DAAL context
	 * @param dataWithLabels - A JavaPairRDD<NumericTable, NumericTable> where
	 * 		the first table is independent variables and the second table is the
	 * 		corresponding known responses.
	 * @return	A Model.
	 */
	public static Model train(
			JavaSparkContext sc,
			DaalContext dc,
			JavaPairRDD<NumericTable, NumericTable> dataWithLabels) {
		final Broadcast<Tuple2<Class<? extends Number>, Method>> config = sc.broadcast(trConfig);
		// Local processing on all slaves 
		JavaRDD<PartialResult> partsrdd = dataWithLabels.map(
				new Function<Tuple2<NumericTable, NumericTable>, PartialResult>() {
					public PartialResult call(Tuple2<NumericTable, NumericTable> tup) {
						DaalContext context = new DaalContext();

    					// Create algorithm to train a model 
						TrainingDistributedStep1Local training = new TrainingDistributedStep1Local(
								context, config.value()._1(), config.value()._2().getMethod());
    					// Set input data on local node 
						tup._1().unpack(context);
						tup._2().unpack(context);
						training.input.set(TrainingInputId.data, tup._1());
						training.input.set(TrainingInputId.dependentVariable, tup._2());
    					// Compute on local node 
    					PartialResult pres = training.compute();
    					pres.pack();
    					
    					context.dispose();
    					return pres;
					}
				}).cache();

    	// Finalizing on master 
    	List<PartialResult> partscollection = partsrdd.collect();
        TrainingDistributedStep2Master master = 
        		new TrainingDistributedStep2Master(dc, trConfig._1(), trConfig._2().getMethod());
        for (PartialResult value : partscollection) {
            value.unpack(dc);
            master.input.add(MasterInputId.partialModels, value);
        }
        master.compute();
        TrainingResult result = master.finalizeCompute();
        return result.get(TrainingResultId.model);
	}
	
	/**
	 * Predict using a pre-trained linear regression model. The prediction is only done
	 * on the master process (Driver program), and no distributed processing is
	 * involved.
	 * @param dc - DAAL context
	 * @param testData - Input dataset
	 * @param model - A pre-trained model
	 * @return A PredictionResult with predicted responses
	 */
	public static PredictionResult predict(
			DaalContext dc,
			NumericTable testData,
			Model model) {
		// Linear Regression prediction only works with batch mode. 
		// Prediction algorithm 
		PredictionBatch predict = new PredictionBatch(dc, predictFpType, predictMethod);
		// Set input 
		predict.input.set(PredictionInputId.data, testData);
		// Set model 
		predict.input.set(PredictionInputId.model, model);
		return predict.compute();
	}
}
