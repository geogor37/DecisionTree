package lab

import toolkit.Matrix
import toolkit.SupervisedLearner
import java.util.*
import kotlin.math.log

class DecisionTree(private val rand: Random): SupervisedLearner(){
	private lateinit var rootNode: Node

	override fun train(features: Matrix, labels: Matrix) {
		rootNode = generateTree(features, labels)
	}

	override fun predict(features: DoubleArray, labels: DoubleArray, target: DoubleArray, isTest: Boolean) {
		TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
	}

	override fun testFinished() {
	}

	private fun generateTree(features: Matrix, labels: Matrix): Node {
		val outputs = labels.data.map { it[0] }
		val uniqueOutputs = outputs.distinct()
		val info = calculateInfo(outputs, uniqueOutputs)
		var bestFeatureIndexSoFar = 0
		var bestInfoGainSoFar = 0.0

		for (colIndex in 0 until features.cols()) {
			if (features.valueCount(colIndex) > 1) {
				val featureValues = features.data.map { it[colIndex] }
				val uniqueFeatureValues = featureValues.distinct()
				var featureInfo = 0.0
				for (value in uniqueFeatureValues) {
					val valueCount = featureValues.count { it == value }
					val filteredOutputs = outputs.filterIndexed { index, _ -> featureValues[index] == value }
					val splitInfo = calculateInfo(filteredOutputs, uniqueOutputs)
					featureInfo += (valueCount / featureValues.size.toDouble()) * splitInfo
				}
				val featureInfoGain = info - featureInfo
				if (featureInfoGain > bestInfoGainSoFar) {
					bestFeatureIndexSoFar = colIndex
					bestInfoGainSoFar = featureInfoGain
				}
			}
		}

		val node = Node(bestFeatureIndexSoFar)
		for (value in features.data.map { it[bestFeatureIndexSoFar] }.distinct()) {
			val filteredLabels = labels.data.filterIndexed { index, _ -> features.get(index, bestFeatureIndexSoFar) == value }
			val filteredFeatures = features.data.filter { it[bestFeatureIndexSoFar] == value }
			val splitLabelMatrix = Matrix().apply { setData(filteredLabels) }
			val splitFeatureMatrix = Matrix().apply { setData(filteredFeatures) }
			if (splitLabelMatrix.valueCount(0) == 1) {
				node.addLeafNode(value, splitLabelMatrix.get(0,0))
			} else {
				node.addLowerLayerNode(value, generateTree(splitFeatureMatrix, splitLabelMatrix))
			}
		}
	}

	private fun calculateInfo(outputs: List<Double>, uniqueOutputs: List<Double>): Double {
		var info = 0.0
		for (output in uniqueOutputs) {
			val outputCount = outputs.count { it == output }
			val ratio = outputCount / outputs.size.toDouble()
			info -= ratio * log(ratio, 2.0)
		}
		return info
	}
}