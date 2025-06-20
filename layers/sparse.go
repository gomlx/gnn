// Package layers implements layers and graph functions that commonly used in GNN operations.
package layers

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
)

// SparseSoftmax takes a logits vector [N] and the indices where they should aggregate [N, 1], and
// returns the probabilities:
//
//	denominator[j] = \sum_{i \in set{indices[i] = j}}(exp(logits[i]))
//	softmask[i] = exp(logits[i]) / denominator[indices[i]]
//
// If logitsMask is not nil, exp(logits[i]) is replaced with 0 where logitsMask is false.
//
// Args:
//   - logits: vector shaped [N] of some float dtype.
//   - logitsMask: nil or a vector shaped [N] of bools.
//   - indices: shapes [N, 1], with the indices on which the logits are part of.
//   - maxIndex: it's the max(indices), it is required statically (not a graph Node) because we reduce
//     sum the exponentials on a temporary node with this shape. It must be known in the graph building time.
//   - sorted: if the indices are sorted set this to true, it will lead to faster runtime in some platforms.
//     In doubt, leave this as false: it could lead to undefined behavior if the indices are not actually sorted.
//
// It returns a vector shaped [N] with the same dtype as logits.
//
// Example:
//
//	logits := Const(g, []float32{ln1, ln2, ln1, ln2, ln1, 0, 0, 0})
//	indices := Const(g, []int32{0, 0, 0, 1, 1, 0, 0, 0})
//	mask := Const(g, []bool{true, true, true, true, true, false, false, false})
//	inputs = []*Node{logits, indices, mask}
//	softmax := SparseSoftmax(logits, mask, ExpandAxes(indices, -1), 2, false)
//	// softmax == []float32{1.0 / 4, 2.0 / 4, 1.0 / 4, 2.0 / 3, 1.0 / 3, 0, 0, 0}
func SparseSoftmax(logits, logitsMask, indices *Node, maxIndex int, sorted bool) *Node {
	tensors.MaxSizeForString = 20
	if !logits.DType().IsFloat() {
		exceptions.Panicf("invalid logits dtype %s, it must be float", logits.DType())
	}
	if logits.Rank() != 1 {
		exceptions.Panicf("invalid logits rank, it must be 1, got shape %s", logits.Shape())
	}
	n := logits.Shape().Dim(0)
	if !indices.DType().IsInt() {
		exceptions.Panicf("invalid indices dtype %s, it must be an int or uint", indices.DType())
	}
	if indices.Shape().CheckDims(n, 1) != nil {
		exceptions.Panicf("indices must be shaped [n=%d, 1], got shape %s", n, indices.Shape())
	}
	if logitsMask != nil {
		if logitsMask.DType() != dtypes.Bool {
			exceptions.Panicf("invalid logitsMask dtype %s, if set it must be a bool", indices.DType())
		}
		if logits.Shape().CheckDims(n) != nil {
			exceptions.Panicf("logitsMask must be shapes [n=%d], got shape %s", n, indices.Shape())
		}
	}

	g := logits.Graph()
	dtype := logits.DType()
	zero := ScalarZero(g, dtype)
	one := ScalarOne(g, dtype)
	minValue := Const(g, dtype.SmallestNonZeroValueForDType())

	// Normalize the logits by subtracting the maximum value: improved numeric stability without any
	// change in the result.
	normalizingMax := BroadcastToDims(minValue, maxIndex)
	tmpLogits := logits
	if logitsMask != nil {
		// We don't want values masked out to participate in the calculation of the max.
		tmpLogits = Where(logitsMask, logits, minValue)
	}
	normalizingMax = ScatterMax(normalizingMax, indices, tmpLogits, sorted, false)
	normalizingMax = Gather(normalizingMax, indices, sorted)
	normalizingMax = StopGradient(normalizingMax)
	normalizedLogits := Sub(logits, normalizingMax)

	// Calculate the numerator for the softmax:
	expLogits := Exp(normalizedLogits)
	if logitsMask != nil {
		expLogits = Where(logitsMask, expLogits, zero)
	}

	// Calculate the denominators:
	sumExpLogits := Zeros(g, shapes.Make(dtype, maxIndex))
	sumExpLogits = ScatterSum(sumExpLogits, indices, expLogits, sorted, false)
	sumExpLogits = Where(Equal(sumExpLogits, zero), one, sumExpLogits) // Avoid division by 0 (NaN) even in masked out values.
	sumExpLogits = Gather(sumExpLogits, indices, sorted)

	// Return the softmax.
	return Div(expLogits, sumExpLogits)
}
