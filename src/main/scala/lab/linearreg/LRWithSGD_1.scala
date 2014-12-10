package lab.linearreg

import org.apache.spark._
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

/* https://spark.apache.org/docs/1.1.0/mllib-linear-methods.html
 */
object LRWithSGD_1 extends App {
  
	val sc = new SparkContext("local", "LRWithSGD_1", System.getenv("SPARK_HOME"))
	
	// Load and parse the data
	val data = sc.textFile("test-data/mllib/ridge-data/lpsa.data")
	
	val parsedData = data.map { line =>
		val parts = line.split(',')
		LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
	}
	
	// Building the model
	val numIterations = 100
	val model = LinearRegressionWithSGD.train(parsedData, numIterations)
	
	// Evaluate model on training examples and compute training error
	val valuesAndPreds = parsedData.map { point =>
	  	val prediction = model.predict(point.features)
	  	(point.label, prediction)
	}
	
	val squaredDiffVP = for(
	    (v,p) <- valuesAndPreds
	) yield math.pow((v - p), 2)
	val MSE = squaredDiffVP.fold(0)(_ + _) // PROBLEM HERE!! NO MEAN
	
	/*val MSE = valuesAndPreds.map( 
	    case (v, p) => math.pow((v - p), 2)
	)//.mean // PROBLEM HERE!! */
	println("training Mean Squared Error = " + MSE)
}