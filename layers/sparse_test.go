package layers

import (
	"math"
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
)

func TestSparseSoftmax(t *testing.T) {
	ln1 := float32(0)
	ln2 := float32(math.Log(2))
	graphtest.RunTestGraphFn(t, "SparseSoftmax", func(g *Graph) (inputs, outputs []*Node) {
		logits := Const(g, []float32{ln1, ln2, ln1, ln2, ln1, 0, 0, 0})
		indices := Const(g, []int32{0, 0, 0, 1, 1, 0, 0, 0})
		mask := Const(g, []bool{true, true, true, true, true, false, false, false})
		inputs = []*Node{logits, indices, mask}
		softmax := SparseSoftmax(logits, mask, ExpandAxes(indices, -1), 2, false)
		outputs = []*Node{softmax}
		return
	}, []any{
		[]float32{1.0 / 4, 2.0 / 4, 1.0 / 4, 2.0 / 3, 1.0 / 3, 0, 0, 0},
	}, 1e-3)
}
