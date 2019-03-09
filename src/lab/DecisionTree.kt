package lab

import toolkit.Matrix
import toolkit.SupervisedLearner
import java.util.*
import kotlin.math.log

class DecisionTree(private val rand: Random): SupervisedLearner(){
	private lateinit var rootNode: Node
	private var treeDepth = 0
	private var nodeCount = 0

	override fun train(features: Matrix, labels: Matrix) {
		treeDepth = 0
		nodeCount = 0
		rootNode = generateTree(1, features.data, labels.data)
		println("Number of nodes: $nodeCount")
		println("Max depth of the tree: $treeDepth")
		var correctPredictions = 0
		features.data.forEachIndexed { index, row ->
			val predictionArray = DoubleArray(1)
			predict(row, predictionArray, DoubleArray(0), false)
			if (labels.data[index][0] == predictionArray[0]) {
				correctPredictions++
			}
		}
		println("Training set accuracy: ${correctPredictions/features.rows().toDouble()}")
	}

	override fun predict(features: DoubleArray, labels: DoubleArray, target: DoubleArray, isTest: Boolean) {
		labels[0] = rootNode.classify(features)
	}

	override fun testFinished() {
	}

	private fun generateTree(currentDepth: Int, features: List<DoubleArray>, labels: List<DoubleArray>): Node {
		if (currentDepth > treeDepth) {
			treeDepth = currentDepth
		}
		val outputs = labels.map { it[0] }
		val uniqueOutputs = outputs.distinct()
		val info = calculateInfo(outputs, uniqueOutputs)
		var bestFeatureIndexSoFar = -1
		var bestInfoGainSoFar = 0.0

		for (colIndex in 0 until features[0].size) {
			val featureValues = features.map { it[colIndex] }
			val uniqueFeatureValues = featureValues.distinct()
			if (uniqueFeatureValues.size > 1) {
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
		nodeCount++
		if (bestFeatureIndexSoFar >= 0) {
			for (value in features.map { it[bestFeatureIndexSoFar] }.distinct()) {
				val filteredLabelArrays = labels.filterIndexed { index, _ -> features[index][bestFeatureIndexSoFar] == value }
				val filteredLabels = filteredLabelArrays.map { it[0] }
				node.defaultOutput = filteredLabels.getMostCommonElement()
				val uniqueLabels = filteredLabels.distinct()
				if (uniqueLabels.size == 1) {
					node.addLeafNode(value, uniqueLabels[0])
					nodeCount++
					if (treeDepth == currentDepth) {
						treeDepth += 1
					}
				} else {
					val filteredFeatures = features.filter { it[bestFeatureIndexSoFar] == value }
					node.addLowerLayerNode(value, generateTree(currentDepth + 1, filteredFeatures, filteredLabelArrays))
				}
			}
		} else {
			node.defaultOutput = labels.map { it[0] }.getMostCommonElement()
		}
		return node
	}

	private fun calculateInfo(outputs: List<Double>, uniqueOutputs: List<Double>): Double {
		var info = 0.0
		for (output in uniqueOutputs) {
			val outputCount = outputs.count { it == output }
			if (outputCount != 0) {
				val ratio = outputCount / outputs.size.toDouble()
				info -= ratio * log(ratio, 2.0)
			}
		}
		return info
	}

	private fun <T> Collection<T>.getMostCommonElement() = groupBy { it }.mapValues { it.value.size }.maxBy { it.value }?.key ?: throw Exception("This shouldn't happen")
}