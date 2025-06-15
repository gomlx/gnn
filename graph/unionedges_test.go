package graph

import (
	"testing"

	"github.com/gomlx/gomlx/types/tensors"
	"github.com/stretchr/testify/require"
)

func TestUnionEdges(t *testing.T) {
	// Test case 1: Basic union with duplicates
	edges1 := tensors.FromValue([][]int32{{0, 1, 0}, {1, 2, 2}})
	edges2 := tensors.FromValue([][]int32{{0, 2}, {1, 3}})

	// Expected result: [[0, 0, 1, 2], [1, 2, 2, 3]] after sorting and removing duplicates
	// The duplicate edge (0,1) from edges2 should be removed.
	// The final list should be sorted.
	expected := [][]int32{{0, 0, 1, 2}, {1, 2, 2, 3}}

	result, err := UnionEdges(edges1, edges2)
	require.NoError(t, err)
	require.NoError(t, SortEdgesBySource(result))
	require.Equal(t, expected, result.Value().([][]int32), "Test Case 1 Failed: Expected %v, got %s", expected, result)

	// Test case 2: No input tensors
	result, err = UnionEdges()
	require.Error(t, err)

	// Test case 4: Single tensor (should just remove internal duplicates)
	edgesWithDupes := tensors.FromValue([][]int32{{1, 0, 1, 0}, {2, 1, 2, 1}})
	expectedSingle := [][]int32{{0, 1}, {1, 2}}
	result, err = UnionEdges(edgesWithDupes)
	require.NoError(t, err)
	require.NoError(t, SortEdgesBySource(result))
	require.Equal(t, expectedSingle, result.Value().([][]int32), "Test Case 4 Failed: Expected %v, got %s", expectedSingle, result)

	// Test case 5: Invalid shape
	invalidShape := tensors.FromValue([]int32{1, 2, 3})
	_, err = UnionEdges(invalidShape)
	require.Error(t, err, "Test Case 5 Failed: Expected error for invalid shape")

	// Test case 6: Invalid dtype
	invalidDType := tensors.FromValue([][]float32{{1, 2}, {3, 4}})
	_, err = UnionEdges(invalidDType)
	require.Error(t, err, "Test Case 6 Failed: Expected error for invalid dtype")
}
