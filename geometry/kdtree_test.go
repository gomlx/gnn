package geometry

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
)

// Helper function to convert a flat slice of points into a slice of "Point" (conceptual) for easier comparison
func convertFlatToConceptualPoints[T KDTreePointType](flatPoints []T, dimension int) [][]T {
	numPoints := len(flatPoints) / dimension
	conceptualPoints := make([][]T, numPoints)
	for i := 0; i < numPoints; i++ {
		conceptualPoints[i] = make([]T, dimension)
		copy(conceptualPoints[i], flatPoints[i*dimension:(i+1)*dimension])
	}
	return conceptualPoints
}

// TestNewKDTree tests the K-d tree construction.
func TestNewKDTree(t *testing.T) {
	// 16 sample 2D points
	originalPointsData := []float64{
		2, 3, // 0
		5, 4, // 1
		9, 6, // 2
		4, 7, // 3
		8, 1, // 4
		7, 2, // 5
		1, 8, // 6
		6, 5, // 7
		10, 10, // 8
		0, 0, // 9
		3, 9, // 10
		11, 2, // 11
		-1, 5, // 12
		12, 8, // 13
		6, 0, // 14
		5, 5, // 15
	}
	dimension := 2
	minPointsPerLeaf := 2

	tree, err := NewKDTree(originalPointsData, dimension, minPointsPerLeaf)
	if err != nil {
		t.Fatalf("Failed to create KD-tree: %v", err)
	}

	if tree == nil {
		t.Fatal("NewKDTree returned a nil tree")
	}
	if tree.NumPoints != len(originalPointsData)/dimension {
		t.Errorf("Incorrect NumPoints. Got %d, want %d", tree.NumPoints, len(originalPointsData)/dimension)
	}
	if tree.Dimension != dimension {
		t.Errorf("Incorrect Dimension. Got %d, want %d", tree.Dimension, dimension)
	}
	if tree.Root == nil {
		t.Fatal("Root node is nil")
	}

	// Log the constructed tree's internal points representation for debugging
	fmt.Println("\n--- Constructed KDTree (internal points representation) ---")
	fmt.Println(tree.String()) // Uses the new KDTree.String() method
	fmt.Println("----------------------------------------------------------")

	// Helper to check node properties
	checkNode := func(t *testing.T, node *KDTreeNode[float64]) {
		if node == nil {
			t.Fatalf("Node is nil when expected a valid node")
			return
		}
		numPointsInNode := tree.NumPointsForNode(node) // Use the helper method
		if numPointsInNode < 0 {
			t.Fatalf("Node has negative number of points: StartIdx %d, EndIdx %d, Dim %d", node.StartIdx, node.EndIdx, tree.Dimension)
		}
		if node.Left == nil && node.Right == nil { // Leaf node
			if numPointsInNode > minPointsPerLeaf {
				t.Fatalf("Node with %d points should not be a leaf (minPointsPerLeaf=%d)", numPointsInNode, minPointsPerLeaf)
			}
			// For leaf nodes, SplitAxis/Value are not meaningful or are default initialized.
		} else { // Internal node
			// Check that left/right nodes are continguous:
			if node.Left.EndIdx != node.Right.StartIdx || node.Right.EndIdx != node.EndIdx || node.Left.StartIdx != node.StartIdx {
				t.Fatalf("Node left/right child indices mismatch. Got left=[%d,%d), right=[%d,%d), current=[%d,%d)",
					node.Left.StartIdx, node.Left.EndIdx, node.Right.StartIdx, node.Right.EndIdx, node.StartIdx, node.EndIdx)
			}

			// Check that the values for the node.SplitAxis on the left are all smaller than the corresponding values on the right.
			if node.Left.Max[node.SplitAxis] >= node.Right.Min[node.SplitAxis] {
				t.Fatalf("Node left/right child split values for axis=%d mismatch. Got left=%v-%v, right=%v-%v, current=%v-%v",
					node.SplitAxis, node.Left.Min, node.Left.Max, node.Right.Min, node.Right.Max, node.Min, node.Max)
			}

			if numPointsInNode <= minPointsPerLeaf {
				t.Errorf("Node with %d points should not be an internal node (minPointsPerLeaf=%d)", numPointsInNode, minPointsPerLeaf)
			}
			// Check if split value correctly partitions points
			for i := node.StartIdx; i < node.EndIdx; i++ { // Iterate over point indices
				pointVal := tree.Points[i*tree.Dimension+node.SplitAxis] // Access point data via flat index
				if i < node.Left.EndIdx && pointVal > node.SplitValue {
					t.Errorf("Point %v (original index %d) in left child conceptual range [%d,%d) but split value %.2f, actual %.2f is greater",
						convertFlatToConceptualPoints(tree.Points[i*tree.Dimension:(i+1)*tree.Dimension], tree.Dimension)[0], tree.Order[i],
						node.Left.StartIdx, node.Left.EndIdx, node.SplitValue, pointVal)
				}
				if i >= node.Right.StartIdx && pointVal < node.SplitValue {
					t.Errorf("Point %v (original index %d) in right child conceptual range [%d,%d) but split value %.2f, actual %.2f is less",
						convertFlatToConceptualPoints(tree.Points[i*tree.Dimension:(i+1)*tree.Dimension], tree.Dimension)[0], tree.Order[i],
						node.Right.StartIdx, node.Right.EndIdx, node.SplitValue, pointVal)
				}
			}
		}

		// Check the bounding box: min <= point <= max for all points in the node's range
		for i := node.StartIdx; i < node.EndIdx; i++ { // Iterate over point indices
			flatIdxStart := i * tree.Dimension
			for d := 0; d < tree.Dimension; d++ {
				pVal := tree.Points[flatIdxStart+d]
				if pVal < node.Min[d] || pVal > node.Max[d] {
					t.Errorf("Point %v (original index %d) at conceptual idx %d is outside node %v bounding box: %v - %v",
						convertFlatToConceptualPoints(tree.Points[flatIdxStart:flatIdxStart+tree.Dimension], tree.Dimension)[0], tree.Order[i], i, node, node.Min, node.Max)
				}
			}
		}
	}

	queue := []*KDTreeNode[float64]{tree.Root}
	nodeCount := 0
	for len(queue) > 0 {
		// Pop the last element from the queue.
		nc := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		nodeCount++
		checkNode(t, nc)
		if nc.Left != nil {
			queue = append(queue, nc.Left)
		}
		if nc.Right != nil {
			queue = append(queue, nc.Right)
		}
	}
	fmt.Printf("\t- successfully checked %d nodes in the tree.\n", nodeCount)

	/*
		---
		### Checking Sorted Points and Order Mapping

		We'll verify that the `Points` slice accurately reflects the tree's structure and that the `Order` slice correctly maps back to the original point positions.

		---
	*/
	t.Run("CheckSortedPointsAndOrder", func(t *testing.T) {
		// Reconstruct original points using KDTree.Order
		reconstructedPoints := make([]float64, len(originalPointsData))
		for i := 0; i < tree.NumPoints; i++ { // Iterate over conceptual point indices
			originalIdx := tree.Order[i] // Get the original index for the point at conceptual index 'i'
			// Copy the point data from its new location in the tree.Points back to its original location
			copy(reconstructedPoints[originalIdx*dimension:(originalIdx+1)*dimension], tree.Points[i*dimension:(i+1)*dimension])
		}

		// Verify that the reconstructed points match the original input
		if !reflect.DeepEqual(reconstructedPoints, originalPointsData) {
			t.Errorf("Reconstructed points do not match original data.\nGot:  %v\nWant: %v", reconstructedPoints, originalPointsData)
		}

		// Additional check: Ensure points within each node's range are indeed sorted by the split axis.
		var verifyNodeSorting func(node *KDTreeNode[float64])
		verifyNodeSorting = func(node *KDTreeNode[float64]) {
			if node == nil || (node.Left == nil && node.Right == nil) { // Leaf node
				return
			}

			// Points in the left child's range
			if node.Left != nil {
				for i := node.Left.StartIdx; i < node.Left.EndIdx; i++ { // Iterate over point indices
					if tree.Points[i*dimension+node.SplitAxis] > node.SplitValue {
						t.Errorf("Point %v (original %d) in left child range [%d,%d) is > split value %.2f on axis %d",
							convertFlatToConceptualPoints(tree.Points[i*dimension:(i+1)*dimension], dimension)[0], tree.Order[i], node.Left.StartIdx, node.Left.EndIdx, node.SplitValue, node.SplitAxis)
					}
				}
				verifyNodeSorting(node.Left)
			}

			// Points in the right child's range
			if node.Right != nil {
				for i := node.Right.StartIdx; i < node.Right.EndIdx; i++ { // Iterate over point indices
					if tree.Points[i*dimension+node.SplitAxis] < node.SplitValue {
						t.Errorf("Point %v (original %d) in right child range [%d,%d) is < split value %.2f on axis %d",
							convertFlatToConceptualPoints(tree.Points[i*dimension:(i+1)*dimension], dimension)[0], tree.Order[i], node.Right.StartIdx, node.Right.EndIdx, node.SplitValue, node.SplitAxis)
					}
				}
				verifyNodeSorting(node.Right)
			}
		}
		verifyNodeSorting(tree.Root)

		fmt.Println("KDTree.Points and KDTree.Order are correctly maintained.")
	})

	/*
		---
		### Edge Cases Testing

		We'll test how the KD-tree handles various edge conditions, such as empty input, 1D data, and scenarios with identical points.

		---
	*/
	t.Run("EdgeCases", func(t *testing.T) {
		// Empty points data
		_, err := NewKDTree([]float64{}, 2, 2)
		require.Error(t, err)

		// dimension=1 (1D points)
		points1D := []float64{10, 5, 20, 15, 2}
		tree1D, err := NewKDTree(points1D, 1, 1)
		if err != nil {
			t.Errorf("NewKDTree for 1D points failed: %v", err)
		}
		if tree1D.Root == nil {
			t.Error("1D tree root is nil")
		}
		if tree1D.Root.SplitAxis != 0 { // Should always be axis 0 for 1D
			t.Errorf("1D tree root split axis mismatch: got %d, want 0", tree1D.Root.SplitAxis)
		}

		// minPointsPerLeaf = 1
		pointsSingleLeaf := []float64{1, 1, 2, 2, 3, 3}
		treeSingleLeaf, err := NewKDTree(pointsSingleLeaf, 2, 1)
		if err != nil {
			t.Errorf("NewKDTree with minPointsPerLeaf=1 failed: %v", err)
		}
		if treeSingleLeaf.Root.Left == nil || treeSingleLeaf.Root.Right == nil {
			t.Error("Tree with minPointsPerLeaf=1 should split more aggressively")
		}
	})

	t.Run("IdenticalPointsOnAxis", func(t *testing.T) {
		pointsIdenticalAxis := []float64{
			1, 10,
			1, 20,
			1, 5,
			1, 15,
		}
		treeIdenticalAxis, err := NewKDTree(pointsIdenticalAxis, 2, 1)
		if err != nil {
			t.Fatalf("Failed for identical points on axis: %v", err)
		}
		// Expect initial split on Y-axis (index 1) as X-axis has range 0.
		if treeIdenticalAxis.Root.SplitAxis != 1 {
			t.Errorf("Expected root split axis to be 1, got %d", treeIdenticalAxis.Root.SplitAxis)
		}
	})

	t.Run("AllIdenticalPoints", func(t *testing.T) {
		pointsAllIdentical := []float64{
			5, 5,
			5, 5,
			5, 5,
			5, 5,
		}
		treeAllIdentical, err := NewKDTree(pointsAllIdentical, 2, 1)
		if err != nil {
			t.Fatalf("Failed for all identical points: %v", err)
		}
		// The root should be a leaf node as no split is possible (maxRange will be 0)
		if treeAllIdentical.Root.Left != nil || treeAllIdentical.Root.Right != nil {
			t.Errorf("Expected root to be a leaf for all identical points, but it split.")
		}
		if treeAllIdentical.NumPointsForNode(treeAllIdentical.Root) != 4 {
			t.Errorf("Leaf node should contain all 4 points, got %d", treeAllIdentical.NumPointsForNode(treeAllIdentical.Root))
		}
	})
}
