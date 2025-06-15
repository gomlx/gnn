package geometry

import (
	"math"

	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// RadiusEdgesConfig is created with RadiusEdges and once fully configured, can be executed
// with Done.
type RadiusEdgesConfig struct {
	source, target *tensors.Tensor
	radius         float64
}

// RadiusEdges returns edges connecting the source to target points that are within the given radius.
//
// This runs only in CPU -- no graphs or backends are used.
//
// Args:
//   - source: shaped [numSourcePoints, dimension], where the dimension is usually 2 or 3.
//     Only float32 and float64 data types are supported.
//   - target: shaped [numTargetPoints, dimension], where the dimension is usually 2 or 3 and must match the source
//     dimension. Same data type as source.
//   - radius: if L2(p_source, p_target) < radius, and edge is created.
//
// It returns a configuration that can be optionally configured. Call RadiusEdgesConfig.Done to perform
// the operation.
// It then returns a tensor "edges" with the shape [2][numEdges]Int32, where edge_i connects
// source point edges[0][i] to target point edges[1][i]. The number of edges (numEdges) varies with the
// points themselves, and if it is not limited, it may be as large as numSourcePoints * numTargetPoints.
//
// TODO: Add MaxNeighbors, batch support, reverting source/target if numTargetPoints >> numSourcePoints.
func RadiusEdges(source, target *tensors.Tensor, radius float64) *RadiusEdgesConfig {
	return &RadiusEdgesConfig{
		source: source,
		target: target,
		radius: radius,
	}
}

// Done performs the RadiusEdges operation as configured.
//
// It then returns a tensor "edges" with the shape [2][numEdges]Int32, where edge_i connects
// source point edges[0][i] to target point edges[1][i]. The number of edges (numEdges) varies with the
// points themselves, and if it is not limited, it may be as large as numSourcePoints * numTargetPoints.
//
// If no edges are found, it returns an error.
func (c *RadiusEdgesConfig) Done() (*tensors.Tensor, error) {
	source := c.source
	target := c.target
	if source.Shape().Rank() != 2 || target.Shape().Rank() != 2 {
		return nil, errors.Errorf("source (%s) and target (%s) must be rank 2: [numPoints, dimension]",
			source.Shape(), target.Shape())
	}
	dimension := source.Shape().Dimensions[1]
	if dimension != target.Shape().Dimensions[1] {
		return nil, errors.Errorf("dimension of the points (last axis) for source (%s) and target (%s) must match",
			source.Shape(), target.Shape())
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
				edgesSource, edgesTarget, err = radiusEdgesImpl(c, flatSource, flatTarget, dimension, float32(c.radius))
			})
		})
	case dtypes.Float64:
		tensors.ConstFlatData[float64](source, func(flatSource []float64) {
			tensors.ConstFlatData[float64](target, func(flatTarget []float64) {
				edgesSource, edgesTarget, err = radiusEdgesImpl(c, flatSource, flatTarget, dimension, c.radius)
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
	if numEdges == 0 {
		return nil, errors.Errorf("no edges found with radius set to %g", c.radius)
	}
	edgesT := tensors.FromShape(shapes.Make(dtypes.Int32, 2, numEdges))
	tensors.MutableFlatData[int32](edgesT, func(flatEdges []int32) {
		copy(flatEdges[:numEdges], edgesSource)
		copy(flatEdges[numEdges:], edgesTarget)
	})
	return edgesT, nil
}

func radiusEdgesImpl[T KDTreePointType](c *RadiusEdgesConfig, source, target []T, dimension int, radius T) (edgesSource, edgesTarget []int32, err error) {
	kd, err := NewKDTree(source, dimension, 16)
	if err != nil {
		return nil, nil, errors.WithMessagef(err, "failed to create KDTree of the source points")
	}

	targetIndices := make([]int32, len(target)/dimension)
	for i := range targetIndices {
		targetIndices[i] = int32(i)
	}
	edgesSource, edgesTarget = radiusEdgesRecursiveImpl(kd, kd.Root, target, targetIndices, dimension, radius, radius*radius, edgesSource, edgesTarget)
	return
}

func radiusEdgesRecursiveImpl[T KDTreePointType](kd *KDTree[T], kdNode *KDTreeNode[T], target []T, targetIndices []int32, dimension int, radius, radius2 T, edgesSource, edgesTarget []int32) ([]int32, []int32) {
	numTargetPoints := len(targetIndices) // == len(target) / dimension

	// Trim target to only those that fit the bounding-box.
	remainingTarget := make([]T, 0, len(target))
	remainingTargetIndices := make([]int32, 0, len(targetIndices))
	for targetPointIdx := range numTargetPoints {
		point := target[targetPointIdx*dimension : (targetPointIdx+1)*dimension]
		if radiusIntersectWithBoundingBox(point, kdNode.Max, kdNode.Min, dimension, radius, radius2) {
			remainingTarget = append(remainingTarget, point...)
			remainingTargetIndices = append(remainingTargetIndices, targetIndices[targetPointIdx])
		}
	}
	if len(remainingTarget) == 0 {
		// No target remains in this split.
		return edgesSource, edgesTarget
	}
	if len(remainingTargetIndices) != len(targetIndices) {
		// Take the selected subset
		target = remainingTarget
		targetIndices = remainingTargetIndices
		numTargetPoints = len(targetIndices) // == len(target) / dimension
	}

	// Stop condition of recursion: for leaf nodes we brute force the remaining target points:
	if kdNode.IsLeaf() {
		for sourcePointIdx := kdNode.StartIdx; sourcePointIdx < kdNode.EndIdx; sourcePointIdx++ {
			for targetPointIdx := range numTargetPoints {
				sourceFlatIdx := sourcePointIdx * dimension
				targetFlatIdx := targetPointIdx * dimension
				dist2 := l2Dist2(kd.Points[sourceFlatIdx:sourceFlatIdx+dimension], target[targetFlatIdx:targetFlatIdx+dimension])
				if dist2 <= radius2 {
					edgesSource = append(edgesSource, int32(kd.Order[sourcePointIdx]))
					edgesTarget = append(edgesTarget, targetIndices[targetPointIdx])
				}
			}
		}
		return edgesSource, edgesTarget
	}

	// Recurse to left and right:
	edgesSource, edgesTarget = radiusEdgesRecursiveImpl(kd, kdNode.Left, target, targetIndices, dimension, radius, radius2, edgesSource, edgesTarget)
	edgesSource, edgesTarget = radiusEdgesRecursiveImpl(kd, kdNode.Right, target, targetIndices, dimension, radius, radius2, edgesSource, edgesTarget)
	return edgesSource, edgesTarget
}

func l2Dist2[T KDTreePointType](a, b []T) T {
	var sum T
	for i, aI := range a {
		diff := aI - b[i]
		sum += diff * diff
	}
	return sum
}

func l2Dist[T KDTreePointType](a, b []T) T {
	return T(math.Sqrt(float64(l2Dist2(a, b))))
}

func radiusIntersectWithBoundingBox[T KDTreePointType](point []T, boundaryMax, boundaryMin []T, dimension int, radius, radius2 T) bool {
	closestPoint := make([]T, dimension)
	for axis := range dimension {
		pAxis := point[axis]
		if pAxis < boundaryMin[axis] {
			if boundaryMin[axis]-pAxis > radius {
				// Optimization: no need to calculate the distance if one axis is already too far.
				return false
			}
			closestPoint[axis] = boundaryMin[axis]
		} else if pAxis > boundaryMax[axis] {
			if pAxis-boundaryMax[axis] > radius {
				// Optimization: no need to calculate the distance if one axis is already too far.
				return false
			}
			closestPoint[axis] = boundaryMax[axis]
		} else {
			closestPoint[axis] = pAxis
		}
	}
	return l2Dist2(point, closestPoint) <= radius2
}
