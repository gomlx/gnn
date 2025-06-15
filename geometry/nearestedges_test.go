package geometry

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

// createRandomPoints creates a tensor of the given shape with random values between -1 and 1.
func createRandomPoints(t *testing.T, numPoints, dimension int, seed uint64) *tensors.Tensor {
	t.Helper()
	if numPoints == 0 {
		return tensors.FromShape(shapes.Make(dtypes.Float32, 0, dimension))
	}
	pointsT := tensors.FromShape(shapes.Make(dtypes.Float32, numPoints, dimension))
	tensors.MutableFlatData(pointsT, func(flat []float32) {
		rng := rand.New(rand.NewPCG(seed, seed+1))
		for i := range flat {
			flat[i] = 2*rng.Float32() - 1
		}
	})
	return pointsT
}

func TestNearestEdges(t *testing.T) {
	const numSourcePoints = 100
	const numTargetPoints = 100
	const dimension = 3

	sourcePointsT := createRandomPoints(t, numSourcePoints, dimension, 42)
	targetPointsT := createRandomPoints(t, numTargetPoints, dimension, 101)

	edgesT, err := NearestEdges(sourcePointsT, targetPointsT).Done()
	require.NoError(t, err)
	require.Equal(t, []int{2, numSourcePoints}, edgesT.Shape().Dimensions, "Edges tensor shape should be [2, numSourcePoints]")

	// Extract data for brute-force verification
	sourcePoints := sourcePointsT.Value().([][]float32)
	targetPoints := targetPointsT.Value().([][]float32)
	edges := edgesT.Value().([][]int32)
	edgesSourceIndices := edges[0]
	edgesTargetIndices := edges[1]

	// Brute-force verification
	for i, sourcePoint := range sourcePoints {
		// Find the closest target point using brute force
		var bruteForceClosestIdx int = -1
		var minDist2 float32 = float32(math.MaxFloat32)

		for j, targetPoint := range targetPoints {
			currentDist2 := l2Dist2(sourcePoint, targetPoint)
			if currentDist2 < minDist2 {
				minDist2 = currentDist2
				bruteForceClosestIdx = j
			}
		}

		// Check if the source index in the edge list is correct
		require.Equal(t, int32(i), edgesSourceIndices[i], "Source index in edge should match the loop index")

		// Check if the found target index matches the brute-force result
		foundTargetIdx := edgesTargetIndices[i]
		require.Equal(t, int32(bruteForceClosestIdx), foundTargetIdx, "For source point %d, expected target %d, but got %d", i, bruteForceClosestIdx, foundTargetIdx)
	}
}
