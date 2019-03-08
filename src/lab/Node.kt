package lab

class Node(private val valueIndex: Int) {
	private val nextLayerMap = mutableMapOf<Double, Node?>()
	private val bottomLayerMap = mutableMapOf<Double, Double>()
	var defaultOutput = 0.0

	// TODO: Make this work when encountering a combination of attributes that doesn't fit the generated tree
	fun classify(instance: DoubleArray): Double = try {
		bottomLayerMap[instance[valueIndex]] ?: nextLayerMap[instance[valueIndex]]?.classify(instance) ?: defaultOutput
	} catch (e: Exception) {
		defaultOutput
	}

	fun addLeafNode(value: Double, output: Double) {
		bottomLayerMap[value] = output
	}

	fun addLowerLayerNode(value: Double, node: Node? = null) {
		nextLayerMap[value] = node
	}
}