package lab

class Node(private val valueIndex: Int) {
	private val nextLayerMap = mutableMapOf<Double, Node?>()
	private val bottomLayerMap = mutableMapOf<Double, Double>()

	fun classify(instance: DoubleArray): Double = bottomLayerMap[instance[valueIndex]] ?: nextLayerMap[instance[valueIndex]]?.classify(instance) ?: -1.0

	fun addLeafNode(value: Double, output: Double) {
		bottomLayerMap[value] = output
	}

	fun addLowerLayerNode(value: Double, node: Node? = null) {
		nextLayerMap[value] = node
	}
}