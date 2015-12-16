package com.intel.daal.spark.rdd.internal.tests;

import org.junit.runner.JUnitCore;
import org.junit.runner.Request;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;

public class DAALTestRunner {

	public static void main(String[] args) throws ClassNotFoundException {
		String[] classAndMethod = args[0].split("#");
		Request request = Request.method(Class.forName(classAndMethod[0]), classAndMethod[1]);
		Result result = new JUnitCore().run(request);
		for (Failure failure : result.getFailures()) {
			System.out.println(failure.toString());
		}
		System.out.println("DONE TESTING");
	}

}

