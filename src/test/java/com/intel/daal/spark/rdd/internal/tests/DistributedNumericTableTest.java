package com.intel.daal.spark.rdd.internal.tests;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.spark.rdd.*;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.mllib.random.RandomRDDs;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import scala.Tuple2;

public class DistributedNumericTableTest {
	
	static JavaSparkContext sc;
	static DaalContext dc;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		SparkConf conf = new SparkConf().setAppName("Spark DAAL test");
		sc = new JavaSparkContext(conf);
		dc = new DaalContext();
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
		sc.stop();
		dc.dispose();
	}

	@Test
	public void fromJavaDoubleRDDShouldProduceDistributedNumericTable() {
		List<Double> data = Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 
										7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
		JavaDoubleRDD distData = sc.parallelizeDoubles(data, 3);
		DistributedNumericTable distNT = DistributedNumericTable.fromJavaDoubleRDD(distData, 2);
		assertEquals("Failed: Correct nCols", 2, distNT.numCols());
		assertEquals("Failed: Correct nRows", 6, distNT.numRows());
		
		JavaDoubleRDD jdRdd = DistributedNumericTable.toJavaDoubleRDD(distNT);
		List<Double> coll = jdRdd.collect();

		double[] all = ArrayUtils.toPrimitive(coll.toArray(new Double[0]));
		double[] orig = ArrayUtils.toPrimitive(data.toArray(new Double[0]));
		assertArrayEquals("Failed: Same arrays", orig, all, 1.0E-7);
	}
	
	@Test
	public void fromRowMatrixShouldProduceDistributedNumericTable() {
		double[] data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
		List<Vector> veclist = new ArrayList<Vector>();
		int ncols = 3;
		int begin = 0; int end = ncols;
		while (begin < data.length) {
			veclist.add(new DenseVector(ArrayUtils.subarray(data, begin, end)));
			begin = end;
			end += ncols;
		}
		
		JavaRDD<Vector> vecrdd = sc.parallelize(veclist, 3);
		RowMatrix matrix = new RowMatrix(vecrdd.rdd());
		DistributedNumericTable distNT = DistributedNumericTable.fromRowMatrix(matrix, 2);
		assertEquals("Failed: Correct nCols", 3, distNT.numCols());
		assertEquals("Failed: Correct nRows", 4, distNT.numRows());

		JavaDoubleRDD jdRdd = DistributedNumericTable.toJavaDoubleRDD(distNT);
		List<Double> coll = jdRdd.collect();

		double[] all = ArrayUtils.toPrimitive(coll.toArray(new Double[0]));
		assertArrayEquals("Failed: Same arrays", data, all, 1.0E-7);
	}
	
	@Test
	public void toJavaDoubleRDDShouldProduceJavaDoubleRDD() {
		List<Double> data = Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 
										7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
		JavaDoubleRDD distData = sc.parallelizeDoubles(data, 3);
		DistributedNumericTable distNT = DistributedNumericTable.fromJavaDoubleRDD(distData, 2);
		JavaDoubleRDD rdd = DistributedNumericTable.toJavaDoubleRDD(distNT);
		assertEquals("Failed: Same number of doubles", data.size(), rdd.count());
		List<Double> coll = rdd.collect();

		double[] orig = ArrayUtils.toPrimitive(data.toArray(new Double[0]));
		double[] actual = ArrayUtils.toPrimitive(coll.toArray(new Double[0]));
		
		assertArrayEquals("Failed: Same arrays", orig, actual, 1.0E-7);
	}
	
	@Test
	public void toRowMatrixShouldProduceRowMatrix() {
		List<Double> data = Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 
										7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
		JavaDoubleRDD distData = sc.parallelizeDoubles(data, 3);
		DistributedNumericTable distNT = DistributedNumericTable.fromJavaDoubleRDD(distData, 2);
		
		RowMatrix matrix = DistributedNumericTable.toRowMatrix(distNT);
		assertEquals("Failed: RowMatrix number of rows", 6, matrix.numRows());
		assertEquals("Failed: RowMatrix number of cols", 2, matrix.numCols());
		
		JavaRDD<Vector> rows = matrix.rows().toJavaRDD();
		List<Vector> coll = rows.collect();
		
		double[] all = new double[0];
		for (Vector row : coll) {
			double[] array = row.toArray();
			all = ArrayUtils.addAll(all, array);
		}
		double[] orig = ArrayUtils.toPrimitive(data.toArray(new Double[0]));
		
		assertArrayEquals("Failed: Same array", orig, all, 1.0E-7);
	}
	
	@Test
	public void pcaCorrTest() {
		final long nrows = 1000;
		final int ncols = 100;
		final int nparts = 3;
		final long seed = 5;
		final int k = 10;

		JavaRDD<Vector> rows = RandomRDDs.normalJavaVectorRDD(sc, nrows, ncols, nparts, seed);
		RowMatrix matrix = new RowMatrix(rows.rdd());
		Matrix mllibres = matrix.computePrincipalComponents(k);
		double[] mllib = mllibres.toArray();
	
		DistributedNumericTable distNT = DistributedNumericTable.randn(sc, nrows, ncols, nparts, seed);
		PCA.PCAResult res = distNT.computePCACorrelationMethod(sc, dc);
		HomogenNumericTable daalres = (HomogenNumericTable) res.loadings;
		double[] daal = ArrayUtils.subarray(daalres.getDoubleArray(), 0, (int)(k*ncols));

		assertArrayEquals("Failed: Same array", mllib, daal, 1.0E-7);
	}
	
	@Test
	public void pcaSvdTest() {
		final long nrows = 1000;
		final int ncols = 100;
		final int nparts = 3;
		final long seed = 5;
		final int k = 10;

		JavaRDD<Vector> rows = RandomRDDs.normalJavaVectorRDD(sc, nrows, ncols, nparts, seed);
		RowMatrix matrix = new RowMatrix(rows.rdd());
		Matrix mllibres = matrix.computePrincipalComponents(k);
		double[] mllib = mllibres.toArray();
	
		DistributedNumericTable distNT = DistributedNumericTable.randn(sc, nrows, ncols, nparts, seed);
		PCA.PCAResult res = distNT.computePCASvdMethod(sc, dc);
		HomogenNumericTable daalres = (HomogenNumericTable) res.loadings;
		double[] daal = ArrayUtils.subarray(daalres.getDoubleArray(), 0, (int)(k*ncols));

		assertArrayEquals("Failed: Same array", mllib, daal, 1.0E-7);
	}
	
	@Test
	public void svdTest() {
		final long nrows = 1000;
		final int ncols = 100;
		final int nparts = 3;
		final long seed = 5;
		final int k = 10;

		JavaRDD<Vector> rows = RandomRDDs.normalJavaVectorRDD(sc, nrows, ncols, nparts, seed);
		RowMatrix matrix = new RowMatrix(rows.rdd());
		SingularValueDecomposition<RowMatrix, Matrix> mllibret = matrix.computeSVD(k, true, 1.0E-9);
		double[] mllibsigma = mllibret.s().toArray();
		
		DistributedNumericTable distNT = DistributedNumericTable.randn(sc, nrows, ncols, nparts, seed);
		SVD.SVDResult daalret = distNT.computeSVD(sc, dc, true);
		HomogenNumericTable daalsigmaNT = (HomogenNumericTable) daalret.sigma;
		double[] daalsigma = ArrayUtils.subarray(daalsigmaNT.getDoubleArray(), 0, k);
		
		assertArrayEquals("Failed: SVD sigma", mllibsigma, daalsigma, 1.0E-7);
		
		RowMatrix mllibU = mllibret.U();
		JavaRDD<Vector> mllibURows = mllibU.rows().toJavaRDD();
		List<Vector> mllibURowsColl = mllibURows.collect();
		
		double[] mllibUFlattened = new double[0];
		for (Vector row : mllibURowsColl) {
			double[] array = row.toArray();
			mllibUFlattened = ArrayUtils.addAll(mllibUFlattened, array);
		}
		
		DistributedNumericTable daalU = daalret.U;
		JavaRDD<NumericTableWithIndex> daalURows = daalU.getTables();
		List<NumericTableWithIndex> daalURowsColl = daalURows.collect();

		double[] daalUFlattened = new double[0];
		for (NumericTableWithIndex t : daalURowsColl) {
			HomogenNumericTable table = (HomogenNumericTable) t.getTable(dc);
			double[] array = table.getDoubleArray();
			daalUFlattened = ArrayUtils.addAll(daalUFlattened, array);
		}
		daalUFlattened = ArrayUtils.subarray(daalUFlattened, 0, (int)(k*nrows));
		
		assertArrayEquals("Failed: SVD U", mllibUFlattened, daalUFlattened, 1.0E-7);
	}
}
