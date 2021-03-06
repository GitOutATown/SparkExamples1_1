package lab.linearreg

import org.apache.spark._
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

/* https://spark.apache.org/docs/1.1.0/mllib-linear-methods.html
 */ 
object SVMWithSGD_2 extends App {
	val sc = new SparkContext("local", "LRWithSGD_2", System.getenv("SPARK_HOME"))
	
	// Load training data in LIBSVM format.
	val data = MLUtils.loadLibSVMFile(sc, "test-data/mllib/sample_libsvm_data.txt")
	println("data.count: " + data.count)
	
	// Split data into training (60%) and test (40%)
	val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
	val training = splits(0).cache()
	val test = splits(1)
	
	// Run training algorithm to build the model
	val numIterations = 100
	val model = SVMWithSGD.train(training, numIterations)
	
	// Clear the default threshold.
	model.clearThreshold()
	
	// Compute raw scores on the test set. 
	val scoreAndLabels = test.map { point =>
		val score = model.predict(point.features)
		(score, point.label)
	}
	
	// Get evaluation metrics.
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	val auROC = metrics.areaUnderROC()
	
	println("Area under ROC = " + auROC)
}



