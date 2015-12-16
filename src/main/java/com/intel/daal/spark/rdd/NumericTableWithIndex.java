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

import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

import scala.Tuple2;

/**
 * Each NumericTableWithIndex is tagged with an index, which is the 
 * offset of its first row within a DistributedNumericTable.
 * 
 * @author Zhang Zhang (zhang.zhang@intel.com)
 *
 */
public class NumericTableWithIndex implements java.io.Serializable {
	
	private Tuple2<Long, NumericTable> tup;
	private long nRows;
	private long nCols;
	
	public NumericTableWithIndex(Long index, NumericTable table) {
		nRows = table.getNumberOfRows();
		nCols = table.getNumberOfColumns();
		table.pack();
		tup = new Tuple2<Long, NumericTable>(index, table);
	}
	
	public NumericTable getTable(DaalContext dc) {
		NumericTable table = tup._2();
		table.unpack(dc);
		return table;
	}

	public Long getIndex() {
		return tup._1();
	}
	
	public long numOfRows() {
		return nRows;
	}
	
	public long numOfCols() {
		return nCols;
	}
	
	public Tuple2<Long, NumericTable> toTuple2() {
		return tup;
	}
}
