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

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.spark.rdd.DistributedNumericTable;
import com.intel.daal.spark.rdd.NumericTableWithIndex;

class TestUtils {
	
	public static void printHomogenNumericTable(String header, HomogenNumericTable table, int maxRows) {
    	StringBuilder builder = new StringBuilder();
    	if (header.length() > 0) {
			builder.append(header);
			builder.append("\n");
    	}
		
    	int ncols = (int) table.getNumberOfColumns();
    	int nrows = (int) table.getNumberOfRows();
    	maxRows = maxRows < nrows ? maxRows : nrows;

		double[] values = table.getDoubleArray();
		
		String dims = String.format("nrows = %d, ncols = %d, nelems = %d \n", nrows, ncols, values.length);
		builder.append(dims);
		for (int i = 0; i < maxRows; ++i) {
			for (int j = 0; j < ncols; ++j) {
				String tmp = String.format("%-6.3f    ", values[i*ncols+j]);
				builder.append(tmp);
			}
			builder.append("\n");
		}
        System.out.println(builder.toString());
	}

	public static void printNumericTableWithIndex(
			DaalContext context,
			String header, 
			NumericTableWithIndex table, 
			int maxRows) {
		System.out.println(header);
		System.out.println("Index = " + table.getIndex());
		printHomogenNumericTable("", (HomogenNumericTable) table.getTable(context), maxRows);
	}
	
	public static void printDistributedNumericTable(
			DaalContext context,
			String header,
			DistributedNumericTable table,
			int maxRows) {
		JavaRDD<Vector> vecRdd  = DistributedNumericTable.toJavaVectorRDD(table);
		List<Vector> veck = vecRdd.take(maxRows);

    	StringBuilder builder = new StringBuilder();
    	builder.append(header);
    	builder.append("\n");
    	
    	for (Vector v : veck) {
    		double[] vals = v.toArray();
    		for (double d : vals) {
    			String tmp = String.format("%-6.3f   ", d);
    			builder.append(tmp);
    		}
            builder.append("\n");
    	}
        System.out.println(builder.toString());
	}
	
    public static void printMLlibResult(String header, Matrix mat, int k) {
    	int nRows = mat.numRows();
    	double[] result = mat.toArray();

        // MLlib result is column-major 
        int resultIndex = 0;
    	StringBuilder builder = new StringBuilder();
    	builder.append(header);
    	builder.append("\n");
        for (long i = 0; i < k; i++) {
            for (long j = 0; j < nRows; j++) {
                String tmp = String.format("%-6.3f   ", result[resultIndex++]);
                builder.append(tmp);
            }
            builder.append("\n");
        }
        System.out.println(builder.toString());
    }

	public static JavaRDD<Vector> loadDataFromFile(JavaSparkContext sc, String path) {
		JavaRDD<String> input = sc.textFile(path);
		return input.map(
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
	}
}
