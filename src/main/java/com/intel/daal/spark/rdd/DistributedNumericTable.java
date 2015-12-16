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
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.DoubleFlatMapFunction;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.mllib.random.RandomRDDs;

import scala.Tuple2;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * Represent an RDD of NumericTableWithIndex. 
 * @author zzhan68
 *
 */
public class DistributedNumericTable implements java.io.Serializable {
	
	private JavaRDD<NumericTableWithIndex> numTables;
	private long nRows;
	private long nCols;
	
	/**
	 * No-arg constructor.
	 */
	public DistributedNumericTable() {
		numTables = null; nRows = 0; nCols = 0;
	}

	/**
	 * Constructor, explicitly specifying dimensions.
	 * @param tables - An RDD of NumericTables.
	 * @param nrows - Total number of rows.
	 * @param ncols - Number of columns.
	 */
	public DistributedNumericTable(
			JavaRDD<NumericTableWithIndex> tables, long nrows, long ncols) {
		numTables = tables;
		nRows = nrows;
		nCols = ncols;
	}
	
	/**
	 * Persist in memory.
	 */
	public void cache() {
		numTables.cache();
	}
	
	/**
	 * @return The RDD of NumericTableWithIndex
	 */
	public JavaRDD<NumericTableWithIndex> getTables() {
		return numTables;
	}

	/**
	 * @return Total number of rows.
	 */
	public long numRows() {
		return nRows;
	}

	/**
	 * @return Number of columns.
	 */
	public long numCols() {
		return nCols;
	}
	
	/**
	 * Cast a DistributedNumericTable to JavaPairRDD<Long, NumericTable>.
	 * @param distNT - The source DistributedNumericTable.
	 * @return A JavaPairRDD<Long, NumericTable> object.
	 */
	public static JavaPairRDD<Long, NumericTable> toJavaPairRDD(DistributedNumericTable distNT) {
		return distNT.numTables.mapToPair(
				new PairFunction<NumericTableWithIndex, Long, NumericTable>() {
					public Tuple2<Long, NumericTable> call(NumericTableWithIndex table) {
						return table.toTuple2();
					}
				});
	}
	
	/**
	 * Split a DistributedNumericTable into a pair JavaPairRDD<NumericTable, NumericTable>.
	 * @param distNT - The DistributedNumericTable to be split. 
	 * @param position - The column index at which to split.
	 * @return A JavaPairRDD that contains a pair of NumericTables.
	 * @throws IllegalArgumentException
	 */
	public static JavaPairRDD<NumericTable, NumericTable> split(
			DistributedNumericTable distNT,
			final int position) throws IllegalArgumentException {
		return distNT.numTables.mapToPair(
				new PairFunction<NumericTableWithIndex, NumericTable, NumericTable>() {
					public Tuple2<NumericTable, NumericTable> call(NumericTableWithIndex nt) {
						DaalContext context = new DaalContext();
						NumericTable table = nt.getTable(context);
						
						double[] data = (table instanceof HomogenNumericTable) ? 
										((HomogenNumericTable) table).getDoubleArray() : null;
						if (data == null) {
							throw new IllegalArgumentException("Invalid NumericTable type");
						}
						long begin = 0; 
						long end = nt.numOfCols();
						double[] first = new double[0];
						double[] second = new double[0];
						
						while (begin < data.length) {
							double[] row = ArrayUtils.subarray(data, (int)begin, (int)end);
							first = ArrayUtils.addAll(
									first, ArrayUtils.subarray(row, 0, position));
							second = ArrayUtils.addAll(
									second, ArrayUtils.subarray(row, position, (int)nt.numOfCols()));
							begin = end;
							end += nt.numOfCols();
						}
						
						HomogenNumericTable t1 = new HomogenNumericTable(
								context, first, position, nt.numOfRows());
						t1.pack();
						HomogenNumericTable t2 = new HomogenNumericTable(
								context, second, nt.numOfCols()-position, nt.numOfRows());
						t2.pack();

						Tuple2<NumericTable, NumericTable> tup = 
								new Tuple2<NumericTable, NumericTable>(t1, t2);
						context.dispose();
						return tup;
					}
				});
	}
	
	/**
	 * Generator method that transforms a JavaRDD<Vector> into a DistributedNumericTable, 
	 * keeping the original partitioning.
	 * @param vecrdd - A JavaRDD<Vector> object.
	 * @param maxRowsPerTable - Max number of rows in eachNumericTable. If this is non-positive
	 * then all rows in a partition are transformed into a single NumericTable. 
	 * @return A DistributedNumericTable backed by HomogenNumericTables.
	 */
	public static DistributedNumericTable fromJavaVectorRDD(
			final JavaRDD<Vector> vecrdd, 
			final int maxRowsPerTable) {
		Vector first = vecrdd.first();
		int ncols = first.size();
		JavaPairRDD<Vector, Long> vecrddWithIds = vecrdd.zipWithIndex();
		
		JavaRDD<NumericTableWithIndex> jntrdd = vecrddWithIds.mapPartitions(
				new FlatMapFunction<Iterator<Tuple2<Vector, Long>>, NumericTableWithIndex>() {
					public List<NumericTableWithIndex> call(Iterator<Tuple2<Vector, Long>> it) {
						DaalContext context = new DaalContext();
						ArrayList<NumericTableWithIndex> tables = new ArrayList<NumericTableWithIndex>();
						int cursize = 0;
						int nrows = 0;
						double[] data = new double[0];
						while (it.hasNext()) {
							Tuple2<Vector, Long> tup = it.next();
							double[] row = tup._1().toArray();
							data = ArrayUtils.addAll(data, row);
							cursize += row.length;
							nrows++;
							if (nrows == maxRowsPerTable || !it.hasNext()) {
								NumericTableWithIndex part = new NumericTableWithIndex(
										tup._2() - nrows + 1,
										new HomogenNumericTable(context, data, cursize/nrows, nrows));
								tables.add(part);
								cursize = 0;
								nrows = 0;
							}
						}
						return tables;
					}
				}, true);
		
		return new DistributedNumericTable(jntrdd, vecrdd.count(), ncols);
	}
	
	/**
	 * Transforms a DistributedNumericTable into a JavaRDD<Vector>.
	 * @param distNT - The source DistributedNumericTable.
	 * @return A JavaRDD<Vector> object.
	 * @throws IllegalArgumentException.
	 */
	public static JavaRDD<Vector> toJavaVectorRDD(
			DistributedNumericTable distNT) throws IllegalArgumentException {
		return distNT.numTables.flatMap(
				new FlatMapFunction<NumericTableWithIndex, Vector>() {
					public List<Vector> call(NumericTableWithIndex nt) {
						DaalContext context = new DaalContext();
						NumericTable table = nt.getTable(context);
						double[] data = (table instanceof HomogenNumericTable) ? 
										((HomogenNumericTable) table).getDoubleArray() : null;
						if (data == null) {
							throw new IllegalArgumentException("Invalid NumericTable type");
						}
						long begin = 0; 
						long end = nt.numOfCols();
						ArrayList<Vector> veclist = new ArrayList<Vector>();
						while (begin < data.length) {
							double[] row = ArrayUtils.subarray(data, (int)begin, (int)end);
							DenseVector dv = new DenseVector(row);
							veclist.add(dv);
							begin = end;
							end += nt.numOfCols();
						}
						context.dispose();
						return veclist;
					}
				});
	}
	
	/**
	 * Generator method that transforms a RowMatrix into a JavaRDD<NumericTable>, keeping the 
	 * original partitioning.
	 * @param matrix - A RowMatrix object as input.
	 * @param maxRowsPerTable - Max number of rows in eachNumericTable. If this is non-positive
	 * then all rows in a partition are transformed into a single NumericTable. 
	 * @return A DistributedNumericTable backed by HomogenNumericTables.
	 */
	public static DistributedNumericTable fromRowMatrix(
			final RowMatrix matrix, final int maxRowsPerTable) {
		JavaRDD<Vector> vecrdd = matrix.rows().toJavaRDD();
		return fromJavaVectorRDD(vecrdd, maxRowsPerTable);
	}
	
	/**
	 * Transforms a DistributedNumericTable into a Spark RowMatrix.
	 * @param distNT - The source DistributedNumericTable.
	 * @return A RowMatrix.
	 */
	public static RowMatrix toRowMatrix(
			final DistributedNumericTable distNT) {
		JavaRDD<Vector> vecrdd = toJavaVectorRDD(distNT);
		return new RowMatrix(vecrdd.rdd());
	}
	
	/**
	 * Transforms a well-formed JavaDoubleRDD into a DistributedNumericTable that is backed
	 * by HomogenNumericTables, with specified number of columns.
	 * @param rdd - The source JavaDoubleRDD.
	 * @param ncols - Number of columns in the result DistributedNumericTable.
	 * @return A DistributedNumericTable.
	 * @throws IllegalArgumentException.
	 */
	public static DistributedNumericTable fromJavaDoubleRDD(
			final JavaDoubleRDD rdd, final int ncols) throws IllegalArgumentException {
		long count = rdd.count();
		if (count/ncols*ncols != count) {
			throw new IllegalArgumentException("Invalid NumericTable dimensions");
		}
		JavaRDD<List<Double>> arrays = rdd.glom();
		JavaRDD<Vector> vecrdd = arrays.flatMap(
				new FlatMapFunction<List<Double>, Vector>() {
					public List<Vector> call(List<Double> dlist) {
						int blocksize = dlist.size();
						Double[] array = dlist.toArray(new Double[blocksize]);
						final double[] unboxed = ArrayUtils.toPrimitive(array);
						ArrayList<Vector> veclist = new ArrayList<Vector>();
						long begin = 0; 
						long end = ncols;
						while (begin < unboxed.length) {
							double[] row = ArrayUtils.subarray(unboxed, (int)begin, (int)end);
							DenseVector dv = new DenseVector(row);
							veclist.add(dv);
							begin = end;
							end += ncols;
						}

						return veclist;
					}
				});

		return fromJavaVectorRDD(vecrdd, 0);
	}
	
	/**
	 * Flattens a DistributedNumericTable backed by HomogenNumericTables into a JavaDoubleRDD.
	 * @param distNT - The source DistributedNumericTable.
	 */
	public static JavaDoubleRDD toJavaDoubleRDD(
			final DistributedNumericTable distNT) {
		JavaDoubleRDD rdd = distNT.numTables.flatMapToDouble(
				new DoubleFlatMapFunction<NumericTableWithIndex>() {
					public List<Double> call(NumericTableWithIndex nt) {
						DaalContext context = new DaalContext();
						NumericTable table = nt.getTable(context);
						if (table instanceof HomogenNumericTable) {
							double[] array = ((HomogenNumericTable) table).getDoubleArray();
							context.dispose();
							return Arrays.asList(ArrayUtils.toObject(array));
						} else {
							context.dispose();
							return null;
						}
					}
				});
		return rdd;
	}
	
	/**
	 * Generator method that generates a DistributedNumericTable backed by HomogenNumericTables
	 * filled with uniformly distributed random numbers. 
	 * @param sc - The JavaSparkContext from which RDD is created.
	 * @param nrows - Total number of rows in the DistributedNumericTable.
	 * @param ncols - Number of columns in the DistributedNumericTable.
	 * @param numPartitions - Number of partitions. If non-positive, use default number of 
	 * 						partitions as determined by sc.
	 * @param seed - RNG seed.
	 * @return A DistributedNumericTable backed by HomogenNumericTables of doubles.
	 */
	public static DistributedNumericTable rand(
			JavaSparkContext sc, long nrows, int ncols, int numPartitions, long seed) {
		if (numPartitions <= 0) {
			numPartitions = sc.defaultParallelism();
		}

		JavaRDD<Vector> rows = RandomRDDs.uniformJavaVectorRDD(sc, nrows, ncols, numPartitions, seed);
		return fromJavaVectorRDD(rows, 0);
	}
	
	/**
	 * Generator method that generates a DistributedNumericTable backed by HomogenNumericTables
	 * filled with normally distributed random numbers. 
	 * @param sc - The JavaSparkContext from which RDD is created.
	 * @param nrows - Total number of rows in the DistributedNumericTable.
	 * @param ncols - Number of columns in the DistributedNumericTable.
	 * @param numPartitions - Number of partitions. If non-positive, use default number of 
	 * 						partitions as determined by sc.
	 * @param seed - RNG seed.
	 * @return A DistributedNumericTable backed by HomogenNumericTables of doubles.
	 */
	public static DistributedNumericTable randn(
			JavaSparkContext sc, long nrows, int ncols, int numPartitions, long seed) {
		if (numPartitions <= 0) {
			numPartitions = sc.defaultParallelism();
		}

		JavaRDD<Vector> rows = RandomRDDs.normalJavaVectorRDD(sc, nrows, ncols, numPartitions, seed);
		return fromJavaVectorRDD(rows, 0);
	}
	
	/**
	 * Compute PCA using the correlation method.
	 * @return Two NumericTables. The first is a 1-by-nCols NumericTable containing all
	 * 		variances sorted in descending order (largest to smallest). The second is a 
	 * 		nCols-by-nCols NumericTable containing all loading vectors in the row-major order. 
	 */
	public PCA.PCAResult computePCACorrelationMethod(JavaSparkContext sc, DaalContext context) {
		PCA.configure(Double.class, PCA.PCAMethod.CORRELATION);
		return PCA.compute(sc, context, numTables);
	}

	/**
	 * Compute PCA using the SVD method. 
	 * @param context - DaalContext.
	 * @return Two NumericTables. The first is a 1-by-nCols NumericTable containing all
	 * 		variances sorted in descending order (largest to smallest). The second is a 
	 * 		nCols-by-nCols NumericTable containing all loading vectors in the row-major order. 
	 */
	public PCA.PCAResult computePCASvdMethod(JavaSparkContext sc, DaalContext context) {
		PCA.configure(Double.class, PCA.PCAMethod.SVD);
		return PCA.compute(sc, context, numTables);
	}
	
	/**
	 * Compute SVD for the distributed NumericTable.
	 * @param context - DaalContext.
	 * @param sc - JavaSparkContext.
	 * @param computeU - Whether to compute U (the left singular matrix).
	 * @return (1) 1-by-nCols NumericTable containing singular values sorted in descending order 
	 * 		(largest to smallest). (2) nCols-by-nCols NumericTable containing the right singular 
	 * 		matrix transposed. (3) nRows-by-nCols distributed NumericTable containing the left
	 * 		singular matrix.
	 */
	public SVD.SVDResult computeSVD(
			JavaSparkContext sc, DaalContext context, boolean computeU) {
		SVD.configure(Double.class, SVD.SVDMethod.DEFAULT);
		return SVD.compute(sc, context, numTables, numRows(), numCols(), computeU);
	}

	/**
	 * Compute QR for the distributed NumericTable.
	 * @param context - DaalContext.
	 * @param sc - JavaSparkContext.
	 * @return (1) nCols-by-nCols NumericTable containing the R1 matrix. 
	 * 		(2) nRows-by-nCols distributed NumericTable containing the Q1 matrix.
	 */
	public QR.QRResult computeQR(
			JavaSparkContext sc, DaalContext context) {
		QR.configure(Double.class, QR.QRMethod.DEFAULT);
		return QR.compute(sc, context, numTables, numRows(), numCols());
	}
	
}
