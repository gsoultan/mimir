package vision

import (
	"cmp"
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// Box is a rectangle in pixels: its top-left corner X1, Y1 and its
// bottom-right corner X2, Y2.
type Box struct{ X1, Y1, X2, Y2 float64 }

// Area returns the box's area, 0 for an empty one.
func (b Box) Area() float64 { return max(b.X2-b.X1, 0) * max(b.Y2-b.Y1, 0) }

// IoU returns the intersection of a and b over their union.
func IoU(a, b Box) float64 {
	w := min(a.X2, b.X2) - max(a.X1, b.X1)
	h := min(a.Y2, b.Y2) - max(a.Y1, b.Y1)
	if w <= 0 || h <= 0 {
		return 0
	}
	inter := w * h
	return inter / (a.Area() + b.Area() - inter)
}

// Detection is a box a detector found, its class and its score.
type Detection struct {
	Box
	Class int
	Score float64
}

// NMS keeps the best-scoring detections and drops each that overlaps a
// kept one of its class by more than iou, as torchvision.ops.batched_nms
// does. It returns the kept ones, best first.
func NMS(dets []Detection, iou float64) []Detection {
	order := slices.Clone(dets)
	slices.SortStableFunc(order, func(a, b Detection) int { return cmp.Compare(b.Score, a.Score) })
	var kept []Detection
	for _, d := range order {
		ok := true
		for _, k := range kept {
			if k.Class == d.Class && IoU(k.Box, d.Box) > iou {
				ok = false
				break
			}
		}
		if ok {
			kept = append(kept, d)
		}
	}
	return kept
}

// BoxIoU returns the IoU of every box of a, [N, 4], with every box of b,
// [M, 4], each x1, y1, x2, y2, as an [N, M] tensor, like
// torchvision.ops.box_iou.
func BoxIoU(a, b *tensor.Tensor) *tensor.Tensor {
	if err := boxes(a, b); err != nil {
		return tensor.Failed(err)
	}
	inter, union := overlap(a.Unsqueeze(1), b.Unsqueeze(0))
	return inter.Div(union)
}

// GIoULoss returns 1 minus the generalized IoU of each predicted box with
// its target, [N, 4] each, averaged, like
// torchvision.ops.generalized_box_iou_loss: unlike 1 - IoU it still
// pulls a box that doesn't overlap its target toward it.
func GIoULoss(pred, target *tensor.Tensor) *tensor.Tensor {
	if err := boxes(pred, target); err != nil {
		return tensor.Failed(err)
	}
	if !slices.Equal(pred.Shape(), target.Shape()) {
		return tensor.Failed(fmt.Errorf("vision: GIoU of %v and %v boxes: %w", pred.Shape(), target.Shape(), tensor.ErrShape))
	}
	inter, union := overlap(pred, target)
	iou := inter.Div(union)
	p, t := corners(pred), corners(target)
	hull := t[2].Maximum(p[2]).Sub(t[0].Minimum(p[0])).Mul(t[3].Maximum(p[3]).Sub(t[1].Minimum(p[1])))
	hull = hull.Maximum(tensor.Scalar[float32](1e-7))
	giou := iou.Sub(hull.Sub(union).Div(hull))
	return tensor.Scalar[float32](1).Sub(giou).Mean()
}

func boxes(ts ...*tensor.Tensor) error {
	for _, t := range ts {
		if err := t.Err(); err != nil {
			return err
		}
		if t.Rank() != 2 || t.Shape()[1] != 4 {
			return fmt.Errorf("vision: boxes %v, want [N, 4]: %w", t.Shape(), tensor.ErrShape)
		}
	}
	return nil
}

// corners splits [..., 4] boxes into x1, y1, x2 and y2.
func corners(b *tensor.Tensor) []*tensor.Tensor {
	s := b.Shape()
	out := make([]*tensor.Tensor, 4)
	for i := range 4 {
		r := make([][2]int, len(s))
		for j, n := range s {
			r[j] = [2]int{0, n}
		}
		r[len(s)-1] = [2]int{i, i + 1}
		out[i] = b.Shrink(r).Squeeze(len(s) - 1)
	}
	return out
}

// overlap returns the intersection and union areas of boxes a and b,
// broadcast against each other.
func overlap(a, b *tensor.Tensor) (inter, union *tensor.Tensor) {
	p, q := corners(a), corners(b)
	zero := tensor.Scalar[float32](0)
	w := p[2].Minimum(q[2]).Sub(p[0].Maximum(q[0])).Maximum(zero)
	h := p[3].Minimum(q[3]).Sub(p[1].Maximum(q[1])).Maximum(zero)
	inter = w.Mul(h)
	area := func(c []*tensor.Tensor) *tensor.Tensor {
		return c[2].Sub(c[0]).Maximum(zero).Mul(c[3].Sub(c[1]).Maximum(zero))
	}
	union = area(p).Add(area(q)).Sub(inter).Maximum(tensor.Scalar[float32](1e-7))
	return inter, union
}

// AveragePrecision returns the mean over classes of the average precision
// of dets against the true boxes, at the IoU threshold iou: mAP@0.5 for
// 0.5, as Pascal VOC scores detectors (with every-point interpolation,
// as COCO's tools). dets[i] and truth[i] are the found and the true boxes
// of picture i. Classes with no true box are left out.
func AveragePrecision(dets, truth [][]Detection, classes int, iou float64) float64 {
	var sum float64
	n := 0
	for c := range classes {
		type found struct {
			pic   int
			score float64
			box   Box
		}
		var all []found
		total := 0
		for p := range truth {
			for _, t := range truth[p] {
				if t.Class == c {
					total++
				}
			}
			if p < len(dets) {
				for _, d := range dets[p] {
					if d.Class == c {
						all = append(all, found{p, d.Score, d.Box})
					}
				}
			}
		}
		if total == 0 {
			continue
		}
		slices.SortStableFunc(all, func(a, b found) int { return cmp.Compare(b.score, a.score) })
		used := map[[2]int]bool{}
		var tp, fp float64
		var recalls, precisions []float64
		for _, f := range all {
			best, at := iou, -1
			for j, t := range truth[f.pic] {
				if t.Class != c || used[[2]int{f.pic, j}] {
					continue
				}
				if v := IoU(f.box, t.Box); v >= best {
					best, at = v, j
				}
			}
			if at >= 0 {
				used[[2]int{f.pic, at}] = true
				tp++
			} else {
				fp++
			}
			recalls = append(recalls, tp/float64(total))
			precisions = append(precisions, tp/(tp+fp))
		}
		// The area under the precision envelope.
		var ap, prev float64
		for i := range recalls {
			best := 0.0
			for _, p := range precisions[i:] {
				best = math.Max(best, p)
			}
			ap += (recalls[i] - prev) * best
			prev = recalls[i]
		}
		sum += ap
		n++
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}
