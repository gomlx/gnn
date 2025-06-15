package graph

import (
	"fmt"
	"sort"

	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
)

func checkEdges(edgesT *tensors.Tensor) error {
	if edgesT.Shape().Rank() != 2 || edgesT.Shape().Dimensions[0] != 2 {
		return fmt.Errorf("invalid shape for edges tensor: got %s, wanted [2, numEdges]", edgesT.Shape())
	}
	if edgesT.DType() != dtypes.Int32 {
		return fmt.Errorf("invalid dtype for edges tensor: got %s, wanted Int32", edgesT.DType())
	}
	return nil
}

// UnionEdges takes a set of edge tensors, combines them, removes duplicates,
// and returns a single tensor with the unique edges.
//
// The input tensors are expected to be of shape [2, numEdges] and have a DType of int32.
// The sorting is done first by the source node index (edges[0]) and then by the target
// node index (edges[1]).
func UnionEdges(inputEdges ...*tensors.Tensor) (*tensors.Tensor, error) {
	if len(inputEdges) == 0 {
		return nil, fmt.Errorf("no input edges provided")
	}

	// Use a map to store unique edges and automatically handle duplicates.
	// The key is a struct representing an edge, which is hashable.
	type edge struct {
		source int32
		target int32
	}
	uniqueEdges := make(map[edge]struct{})
	var empty struct{}

	for _, edgesT := range inputEdges {
		if edgesT == nil || edgesT.Shape().Size() == 0 {
			continue
		}
		err := checkEdges(edgesT)
		if err != nil {
			return nil, err
		}

		numEdges := edgesT.Shape().Dimensions[1]
		edgesData := edgesT.Value().([][]int32)
		sourceNodes := edgesData[0]
		targetNodes := edgesData[1]

		for i := 0; i < numEdges; i++ {
			uniqueEdges[edge{source: sourceNodes[i], target: targetNodes[i]}] = empty
		}
	}

	if len(uniqueEdges) == 0 {
		return tensors.FromShape(shapes.Make(dtypes.Int32, 2, 0)), nil
	}

	// Create the final output tensor.
	numUniqueEdges := len(uniqueEdges)
	outputShape := shapes.Make(dtypes.Int32, 2, numUniqueEdges)
	outputTensor := tensors.FromShape(outputShape)
	tensors.MutableFlatData(outputTensor, func(flat []int32) {
		var edgeIdx int
		for e := range uniqueEdges {
			flat[edgeIdx] = e.source
			flat[edgeIdx+numUniqueEdges] = e.target
			edgeIdx++
		}
	})
	return outputTensor, nil
}

// SortEdgesBySource in-place in the tensor. The tensor contents are mutated -- and moved to local storage if
// they were stored in an accelerator before.
func SortEdgesBySource(edges *tensors.Tensor) error {
	err := checkEdges(edges)
	if err != nil {
		return err
	}
	tensors.MutableFlatData(edges, func(flat []int32) {
		sort.Sort(edgesSortableBySource(flat))
	})
	return nil
}

type edgesSortableBySource []int32

func (edges edgesSortableBySource) Len() int { return len(edges) / 2 }
func (edges edgesSortableBySource) Less(i, j int) bool {
	// Check source ids.
	if edges[i] != edges[j] {
		return edges[i] < edges[j]
	}
	numEdges := edges.Len()
	// Secondary order is by target ids.
	return edges[i+numEdges] < edges[j+numEdges]
}
func (edges edgesSortableBySource) Swap(i, j int) {
	numEdges := edges.Len()
	edges[i], edges[j] = edges[j], edges[i]                                     // source ids
	edges[i+numEdges], edges[j+numEdges] = edges[j+numEdges], edges[i+numEdges] // target ids
}
