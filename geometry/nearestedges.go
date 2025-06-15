package geometry

import (
	"math"

	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// NearestEdgesConfig is created with NearestEdges and once fully configured, can be executed
// with Done.
type NearestEdgesConfig struct {
	source, target *tensors.Tensor
}

// NearestEdges returns edges connecting each source point to its closest target point.
//
// This runs only on CPU -- no graphs or backends are used.
//
// Args:
//   - source: shaped [numSourcePoints, dimension], where the dimension is usually 2 or 3.
//     Only float32 and float64 data types are supported.
//   - target: shaped [numTargetPoints, dimension], where the dimension must match the source
//     dimension. Same data type as source.
//
// It returns a configuration that can be optionally configured. Call NearestEdgesConfig.Done to perform
// the operation.
// It then returns a tensor "edges" with the shape [2, numSourcePoints]Int32, where edge_i connects
// source point edges[0][i] to target point edges[1][i].
func NearestEdges(source, target *tensors.Tensor) *NearestEdgesConfig {
	return &NearestEdgesConfig{
		source: source,
		target: target,
	}
}

// Done performs the NearestEdges operation as configured.
//
// It returns a tensor "edges" with the shape [2, numSourcePoints]Int32, where edge_i connects
// source point i to its closest target point.
//
// It is an error if there are no target points.
func (c *NearestEdgesConfig) Done() (*tensors.Tensor, error) {
	source := c.source
	target := c.target
	if source == nil || target == nil || source.Size() == 0 || target.Size() == 0 {
		return nil, errors.Errorf("nearest edges source(%s) or target(%s) are empty",
			source.Shape(), target.Shape())
	}
	if source.Shape().Rank() != 2 || target.Shape().Rank() != 2 {
		return nil, errors.Errorf("source (%s) and target (%s) must be rank 2: [numPoints, dimension]",
			source.Shape(), target.Shape())
	}
	dimension := source.Shape().Dimensions[1]
	if dimension != target.Shape().Dimensions[1] {
		return nil, errors.Errorf("dimension of the points (last axis) for source (%s) and target (%s) must match",
			source.Shape(), target.Shape())
	}
	if target.Shape().Dimensions[0] == 0 {
		return nil, errors.Errorf("target tensor cannot be empty")
	}
	dtype := source.DType()
	if dtype != target.DType() {
		return nil, errors.Errorf("DType of the source (%s) and target (%s) must match and be either Float32 or Float64",
			source.Shape(), target.Shape())
	}

	var edgesSource, edgesTarget []int32
	var err error
	switch dtype {
	case dtypes.Float32:
		tensors.ConstFlatData[float32](source, func(flatSource []float32) {
			tensors.ConstFlatData[float32](target, func(flatTarget []float32) {
				edgesSource, edgesTarget, err = nearestEdgesImpl(c, flatSource, flatTarget, dimension, math.MaxFloat32)
			})
		})
	case dtypes.Float64:
		tensors.ConstFlatData[float64](source, func(flatSource []float64) {
			tensors.ConstFlatData[float64](target, func(flatTarget []float64) {
				edgesSource, edgesTarget, err = nearestEdgesImpl(c, flatSource, flatTarget, dimension, math.MaxFloat64)
			})
		})
	default:
		return nil, errors.Errorf("DType of the source (%s) and target (%s) must match and be either Float32 or Float64",
			source.Shape(), target.Shape())
	}
	if err != nil {
		return nil, err
	}
	numEdges := len(edgesSource)
	if len(edgesTarget) != numEdges {
		return nil, errors.Errorf("edges number of source indices (%d) different from the number of target indices (%d)!? something is wrong in the algorithm, or some cosmic ray hit the server",
			numEdges, len(edgesTarget))
	}
	if numEdges != source.Shape().Dimensions[0] {
		return nil, errors.Errorf("number of edges (%d) != number of source points (%d)!? something is wrong in the algorithm, or some cosmic ray hit the server",
			numEdges, source.Shape().Dimensions[0])
	}

	edgesT := tensors.FromShape(shapes.Make(dtypes.Int32, 2, numEdges))
	tensors.MutableFlatData[int32](edgesT, func(flatEdges []int32) {
		copy(flatEdges[:numEdges], edgesSource)
		copy(flatEdges[numEdges:], edgesTarget)
	})
	return edgesT, nil
}

func nearestEdgesImpl[T KDTreePointType](_ *NearestEdgesConfig, source, target []T, dimension int, maxValue T) (edgesSource, edgesTarget []int32, err error) {
	// Build KD-tree on target points for efficient search.
	kd, err := NewKDTree(target, dimension, 16)
	if err != nil {
		return nil, nil, errors.WithMessagef(err, "failed to create KDTree of the target points")
	}

	numSourcePoints := len(source) / dimension
	edgesSource = make([]int32, numSourcePoints)
	edgesTarget = make([]int32, numSourcePoints)

	for i := range numSourcePoints {
		sourcePoint := source[i*dimension : (i+1)*dimension]
		bestTargetIdx := findNearest(kd, sourcePoint, maxValue)
		edgesSource[i] = int32(i)
		edgesTarget[i] = int32(bestTargetIdx)
	}

	return
}

// findNearest searches the kd-tree for the nearest neighbor to the given point.
// It returns the original index of the nearest point and the squared distance.
func findNearest[T KDTreePointType](kd *KDTree[T], point []T, maxValue T) int32 {
	best := &nearestBestMatch[T]{
		dist2: maxValue,
		index: -1,
	}
	findNearestRecursive(kd, kd.Root, point, best)
	return int32(kd.Order[best.index])
}

type nearestBestMatch[T KDTreePointType] struct {
	index int
	dist2 T
}

func findNearestRecursive[T KDTreePointType](kd *KDTree[T], node *KDTreeNode[T], point []T, best *nearestBestMatch[T]) {
	if node == nil {
		return
	}

	// If it's a leaf node, brute force check all points in it
	if node.IsLeaf() {
		for i := node.StartIdx; i < node.EndIdx; i++ {
			dist2 := l2Dist2(point, kd.Points[i*kd.Dimension:(i+1)*kd.Dimension])
			if dist2 < best.dist2 {
				best.dist2 = dist2
				best.index = i
			}
		}
		return
	}

	// Recurse down the tree
	var first, second *KDTreeNode[T]
	if point[node.SplitAxis] < node.SplitValue {
		first, second = node.Left, node.Right
	} else {
		first, second = node.Right, node.Left
	}

	// Go down the most promising branch first
	findNearestRecursive[T](kd, first, point, best)

	// Check if we need to check the other branch.
	// We only need to if the distance from the point to the other branch's bounding box
	// is less than our current best distance.
	distToSplit := point[node.SplitAxis] - node.SplitValue
	distToSplit2 := distToSplit * distToSplit

	if distToSplit2 < best.dist2 {
		findNearestRecursive[T](kd, second, point, best)
	}
}
