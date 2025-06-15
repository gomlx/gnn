package geometry

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

func TestRadiusEdges(t *testing.T) {
	// Create random points uniformly distributed in the range [-1, -1] - [0, 0].
	// The target points are the 5 following centroids:
	targetPointsT := tensors.FromValue([][]float32{
		{0, 0},
		{-0.5, -0.5},
		{-0.5, 0.5},
		{0.5, -0.5},
		{0.5, 0.5}})
	numSourcePoints := 1000
	sourcePointsT := tensors.FromShape(shapes.Make(dtypes.Float32, numSourcePoints, 2))
	tensors.MutableFlatData(sourcePointsT, func(flat []float32) {
		rng := rand.New(rand.NewPCG(0, 42))
		for i := range flat {
			flat[i] = 2*rng.Float32() - 1
		}
	})

	const radius = 0.3
	edgesT, err := RadiusEdges(sourcePointsT, targetPointsT, radius).Done()
	require.NoError(t, err)

	sourcePoints := sourcePointsT.Value().([][]float32)
	targetPoints := targetPointsT.Value().([][]float32)
	edges := edgesT.Value().([][]int32)
	edgesSourceIndices := edges[0]
	edgesTargetIndices := edges[1]

	// Track seen edges to check for duplicates
	seen := make(map[string]bool)
	for i := range edgesSourceIndices {
		edge := fmt.Sprintf("%d-%d", edgesSourceIndices[i], edgesTargetIndices[i])
		require.False(t, seen[edge], "Found duplicate edge: source=%d, target=%d",
			edgesSourceIndices[i], edgesTargetIndices[i])
		seen[edge] = true
	}

	// Verify that all connected points are within radius distance
	for i := range edgesSourceIndices {
		sourcePoint := sourcePoints[edgesSourceIndices[i]]
		targetPoint := targetPoints[edgesTargetIndices[i]]
		dist := l2Dist(sourcePoint, targetPoint)
		require.LessOrEqual(t, dist, float32(radius), "Distance between connected points should be <= radius")
	}

	// Brute-force count of all point pairs within radius distance
	pairsCount := 0
	for i := range sourcePoints {
		for j := range targetPoints {
			if l2Dist(sourcePoints[i], targetPoints[j]) <= radius {
				edgeStr := fmt.Sprintf("%d-%d", i, j)
				if !seen[edgeStr] {
					fmt.Printf("Edge %s not seen: source=%v, target=%v, distance=%.2g\n",
						edgeStr, sourcePoints[i], targetPoints[j], l2Dist(sourcePoints[i], targetPoints[j]))
				}
				pairsCount++
			}
		}
	}

	require.Equal(t, pairsCount, len(edgesSourceIndices),
		"Number of edges should match number of point pairs within radius distance")
}

func makeGridPoints(n int) [][]float32 {
	points := make([][]float32, n*n*n)
	idx := 0
	for x := 0; x < n; x++ {
		for y := 0; y < n; y++ {
			for z := 0; z < n; z++ {
				// Convert from [0,n-1] to [-1,1] range
				points[idx] = []float32{
					2*float32(x)/float32(n-1) - 1,
					2*float32(y)/float32(n-1) - 1,
					2*float32(z)/float32(n-1) - 1,
				}
				idx++
			}
		}
	}
	return points
}

func BenchmarkRadiusEdges(b *testing.B) {
	// Setup source points
	numSourcePoints := 100_000
	sourcePointsT := tensors.FromShape(shapes.Make(dtypes.Float32, numSourcePoints, 3))
	tensors.MutableFlatData(sourcePointsT, func(flat []float32) {
		rng := rand.New(rand.NewPCG(0, 42))
		for i := range flat {
			flat[i] = 2*rng.Float32() - 1
		}
	})

	// Setup target points as 32x32x32 grid
	gridSize := 32
	targetPointsT := tensors.FromValue(makeGridPoints(gridSize))

	const radius = math.Sqrt2 * 1.0 / 32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := RadiusEdges(sourcePointsT, targetPointsT, radius).Done()
		if err != nil {
			b.Fatal(err)
		}
	}
}
