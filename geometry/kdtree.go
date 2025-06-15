package geometry

import (
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/pkg/errors"
)

// KDTreePointType accepted by KDTree.
type KDTreePointType interface{ float32 | float64 }

// KDTree is a structured container of NumPoints points with the given Dimension, organized as a binary tree.
//
// It's a convenient structure to quickly search for points in areas of the space.
//
// See NewKDTree to construct the kd-tree.
type KDTree[T KDTreePointType] struct {
	// Points has size NumPoints * Dimension, the underlying shape being [NumPoints, Dimension], stored in row-major order.
	// This slice is modified in-place to reflect the KD-tree's sorting.
	Points []T

	// NumPoints stored in the KD-tree.
	NumPoints int

	// Dimension of each point.
	Dimension int

	// Order maps each index on KDTree.Points (conceptual point index) to the corresponding point index in original pointData.
	// So if KDTree.Points[i*Dimension:(i+1)*Dimension] corresponds to the original pointData[j*Dimension:(j+1)*Dimension], we have Order[i] = j.
	// len(Order) == NumPoints.
	// This slice is also modified in-place during tree construction.
	Order []int

	// Root of the tree.
	Root *KDTreeNode[T]
}

// KDTreeNode represents a node in the K-d tree.
type KDTreeNode[T KDTreePointType] struct {
	// Min coordinates for the bounding box of this node's region.
	// len(Min) == KDTree.Dimension.
	Min []T

	// Max coordinates for the bounding box of this node's region
	// len(Max) == KDTree.Dimension.
	Max []T

	// StartIdx is the index of the first point (in KDTree.Points and KDTree.Order) included in this node.
	StartIdx int

	// EndIdx is the one-past index of the last point (in KDTree.Points and KDTree.Order) included in this node.
	EndIdx int // End index (exclusive) in the sorted points slice

	// Left And Right if not a leaf node.
	Left, Right *KDTreeNode[T]

	// SplitAxis for this node, if not a leaf node.
	SplitAxis int

	// SplitValue for this node, if not a leaf node.
	// Points with the SplitAxis value < SplitValue go to the Left node. Otherwise, they go to the Right node.
	SplitValue T
}

// NumPointsForNode contained inside the bounding box of the node.
func (tree *KDTree[T]) NumPointsForNode(node *KDTreeNode[T]) int {
	// With StartIdx/EndIdx as point indices, this is now direct.
	return node.EndIdx - node.StartIdx
}

// IsLeaf node.
func (node *KDTreeNode[T]) IsLeaf() bool {
	return node.Left == nil && node.Right == nil
}

// NewKDTree builds a K-d tree from a flat slice of point values.
// The splits are chosen on the axis with the largest range, and they take the median point for the axis
// to keep the generated tree approximately balanced.
//
// Args:
//   - pointsData: A flat slice of float64 where points are laid out contiguously
//     (e.g., [x1, y1, z1, x2, y2, z2, ...]). It will be cloned and sorted into KDTree.Points.
//   - D: The number of axes for each point (dimension of a point).
//   - minPointsPerLeaf: The minimum number of points a node must contain to be split further.
//
// It is an error to provide 0 points.
func NewKDTree[T KDTreePointType](pointsData []T, dimension int, minPointsPerLeaf int) (*KDTree[T], error) {
	if len(pointsData) == 0 {
		return nil, errors.Errorf("NewKDTree with empty pointsData")
	}
	fmt.Printf("NewKDTree[%T](dimension=%d, minPointsPerLeaf=%d)\n", pointsData[0], dimension, minPointsPerLeaf)
	if dimension <= 0 {
		return nil, errors.Errorf("number of dimensions (dimension) must be positive")
	}
	if len(pointsData)%dimension != 0 {
		return nil, errors.Errorf("length of pointsData (%d) must be a multiple of the dimension of each point (%d)", len(pointsData), dimension)
	}
	if minPointsPerLeaf < 1 {
		return nil, errors.Errorf("minPointsPerLeaf must be at least 1")
	}

	numPoints := len(pointsData) / dimension
	if numPoints == 0 {
		return &KDTree[T]{NumPoints: 0, Dimension: dimension}, nil // Empty tree for no points
	}

	// Initialize the Order slice to map original indices.
	order := make([]int, numPoints)
	for i := 0; i < numPoints; i++ {
		order[i] = i
	}

	tree := &KDTree[T]{
		Points:    slices.Clone(pointsData),
		NumPoints: numPoints,
		Dimension: dimension,
		Order:     order, // This will also be reordered during sorting.
	}

	// Calculate initial bounding box for the entire point set
	// This still needs to iterate over all points in their original layout to find global min/max

	// Start building the tree recursively
	// Now, start/end are point indices directly.
	tree.Root = tree.buildNode(0, tree.NumPoints, minPointsPerLeaf)

	return tree, nil
}

// buildNode recursively constructs a KDTreeNode.
// startPointIdx, endPointIdx: The range of point indices (in a tree.Points and tree.Order) that this node represents.
// minCoords, maxCoords: The bounding box for the current node's region.
func (tree *KDTree[T]) buildNode(startPointIdx, endPointIdx int, minPointsPerLeaf int) *KDTreeNode[T] {
	//fmt.Printf("> buildNode(%d,%d), numPointsInNode=%d\n", startPointIdx, endPointIdx, endPointIdx-startPointIdx)
	dimension := tree.Dimension
	numPointsInNode := endPointIdx - startPointIdx
	minCoords, maxCoords := calculateBoundingBox(tree.Points[startPointIdx*dimension:endPointIdx*dimension], dimension)
	node := &KDTreeNode[T]{
		Min:      minCoords,
		Max:      maxCoords,
		StartIdx: startPointIdx, // StartIdx is now a point index
		EndIdx:   endPointIdx,   // EndIdx is now a point index
	}

	if numPointsInNode <= minPointsPerLeaf {
		// This node is a leaf
		return node
	}

	// 1. Find the axis with the largest range for points within this node's bounding box
	splitAxis := -1
	var maxRange T = -1.0 // Initialize with a negative value for float

	for axis := 0; axis < dimension; axis++ {
		currentRange := maxCoords[axis] - minCoords[axis]
		if currentRange > maxRange {
			maxRange = currentRange
			splitAxis = axis
		}
	}

	// If all points in this node are identical (range is 0 for all axes), we can't split further.
	// This can happen if numPointsInNode > minPointsPerLeaf but all points are at the same coordinate.
	if maxRange == 0 {
		return node // Treat as a leaf
	}

	node.SplitAxis = splitAxis

	// 2. Sort the relevant portion of the points (and their original order) along the chosen axis:
	order := make([]int, numPointsInNode)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		flatIdxI := (order[i]+startPointIdx)*dimension + splitAxis
		flatIdxJ := (order[j]+startPointIdx)*dimension + splitAxis
		return tree.Points[flatIdxI] < tree.Points[flatIdxJ]
	})

	// Update the tree.Order and tree.Points accordingly, using temporary slices to store the reordered data
	tempPoints := make([]T, numPointsInNode*dimension)
	tempOrder := make([]int, numPointsInNode)

	// Reorder the points and their original indices based on the sorting order
	for dstIdx, srcIdx := range order {
		srcPointIdx := srcIdx + startPointIdx
		// Copy point coordinates
		copy(tempPoints[dstIdx*dimension:(dstIdx+1)*dimension],
			tree.Points[srcPointIdx*dimension:(srcPointIdx+1)*dimension])
		// Copy original point index
		tempOrder[dstIdx] = tree.Order[srcPointIdx]
	}

	// Copy the reordered data back to the original slices
	copy(tree.Points[startPointIdx*tree.Dimension:endPointIdx*dimension], tempPoints)
	copy(tree.Order[startPointIdx:endPointIdx], tempOrder)

	// 3. Find the median split point
	medianPointIdx := startPointIdx + numPointsInNode/2

	// The median value is now directly accessible from the sorted `tree.Points` at the `medianPointIdx`
	// (converted to flat index).
	node.SplitValue = tree.Points[medianPointIdx*dimension+splitAxis]

	// Adjust medianPointIdx if there are duplicate split values at the median.
	// We want the left child to contain all points with SplitAxis value < SplitValue,
	// and the right child to contain all points with SplitAxis value >= SplitValue.
	// So, we find the first index that has the SplitValue, effectively putting all
	// points with the split value or greater into the right child.
	for medianPointIdx > startPointIdx && tree.Points[(medianPointIdx-1)*dimension+splitAxis] >= node.SplitValue {
		medianPointIdx--
	}
	if medianPointIdx == startPointIdx {
		// Degenerate case where there are too many ties on one axis: we simply don't split for now
		// TODO: attempt split on other axes.
		return node
	}

	// Recursively build left and right children
	// Left child: points from 'startPointIdx' up to 'medianPointIdx' (exclusive)
	if medianPointIdx > startPointIdx {
		node.Left = tree.buildNode(startPointIdx, medianPointIdx, minPointsPerLeaf)
	}

	// Right child: points from 'medianPointIdx' up to 'endPointIdx' (exclusive)
	if medianPointIdx < endPointIdx {
		node.Right = tree.buildNode(medianPointIdx, endPointIdx, minPointsPerLeaf)
	}
	return node
}

// calculateBoundingBox determines the min and max coordinates for a given set of points.
// It iterates over the input `pointsData` (which is the full flat slice) to find the global min/max.
func calculateBoundingBox[T KDTreePointType](pointsData []T, dimension int) ([]T, []T) {
	minCoords := make([]T, dimension)
	maxCoords := make([]T, dimension)
	numPoints := len(pointsData) / dimension

	if numPoints == 0 {
		return minCoords, maxCoords // Return empty slices for no points
	}

	// Initialize with the first point's values
	for d := 0; d < dimension; d++ {
		minCoords[d] = pointsData[d]
		maxCoords[d] = pointsData[d]
	}

	// Iterate through the rest of the points to update min/max
	for i := 1; i < numPoints; i++ {
		for d := 0; d < dimension; d++ {
			pointVal := pointsData[i*dimension+d]
			if pointVal < minCoords[d] {
				minCoords[d] = pointVal
			}
			if pointVal > maxCoords[d] {
				maxCoords[d] = pointVal
			}
		}
	}
	return minCoords, maxCoords
}

// String implements the fmt.Stringer interface for KDTree, providing a hierarchical
// representation of the tree structure.
func (tree *KDTree[T]) String() string {
	if tree == nil {
		return "nil KDTree"
	}
	if tree.NumPoints == 0 {
		return "KDTree: No points"
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("KDTree (NumPoints: %d, Dimension: %d):\n", tree.NumPoints, tree.Dimension))
	sb.WriteString("-------------------------------------------\n")
	tree.stringifyNode(&sb, "Root", tree.Root, 0)
	sb.WriteString("-------------------------------------------\n")
	return sb.String()
}

// stringifyNode recursively prints the node and its children with proper indentation.
func (tree *KDTree[T]) stringifyNode(sb *strings.Builder, prefix string, node *KDTreeNode[T], depth int) {
	indent := strings.Repeat("  ", depth)

	// Print node information
	if node.Left != nil || node.Right != nil {
		sb.WriteString(fmt.Sprintf("%s%s node (axis: %d, value: %.2f, bounding-box=%v - %v):\n", indent, prefix, node.SplitAxis, node.SplitValue, node.Min, node.Max))
		tree.stringifyNode(sb, "Left", node.Left, depth+1)
		tree.stringifyNode(sb, "Right", node.Right, depth+1)
		return
	}

	// Leaf node:
	sb.WriteString(fmt.Sprintf("%s%s leaf Node:\n", indent, prefix))

	// Print points in this node
	for i := node.StartIdx; i < node.EndIdx; i++ {
		pointStart := i * tree.Dimension
		pointEnd := pointStart + tree.Dimension
		point := tree.Points[pointStart:pointEnd]
		originalIdx := tree.Order[i]

		sb.WriteString(indent + "  [")
		for d := 0; d < tree.Dimension; d++ {
			sb.WriteString(fmt.Sprintf("%.2g", point[d]))
			if d < tree.Dimension-1 {
				sb.WriteString(", ")
			}
		}
		sb.WriteString(fmt.Sprintf("] (Original Index: %d)\n", originalIdx))
	}
}
