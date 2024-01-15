package arcface

import (
	"fmt"
	"sort"

	"github.com/ivansuteja96/go-onnxruntime"
)

const (
	nmsThresh = float32(0.4)
	detThresh = float32(0.5)
)

var (
	// len(outputs)==9
	_fmc           = 3
	_featStrideFpn = []int{8, 16, 32}
	_numAnchors    = 2
)

// process result after face-detect model inferenced
func processResult(netOuts []onnxruntime.TensorValue, detScale float32) ([][]float32, [][]float32) {
	//for i:=0;i<len(net_outs);i++ {
	//	log.Printf("Success do predict, shape : %+v, result : %+v\n",
	//		net_outs[i].Shape,
	//		net_outs[i].Value.([]float32)[:net_outs[i].Shape[1]], // only show one value
	//	)
	//}

	centerCache := make(map[string][][]float32)

	var scoresList []float32
	var bboxesList [][]float32
	var kpssList [][]float32

	for idx := range _featStrideFpn {
		stride := _featStrideFpn[idx]
		scores := netOuts[idx].Value.([]float32)
		bboxPreds := netOuts[idx+_fmc].Value.([]float32)
		for i := range bboxPreds {
			bboxPreds[i] = bboxPreds[i] * float32(stride)
		}
		// landmark
		kpsPreds := netOuts[idx+_fmc*2].Value.([]float32)
		for i := range kpsPreds {
			kpsPreds[i] = kpsPreds[i] * float32(stride)
		}

		height := detModelInputSize / stride
		width := detModelInputSize / stride
		key := fmt.Sprintf("%d-%d-%d", height, width, stride)
		var anchorCenters [][]float32
		if val, ok := centerCache[key]; ok {
			anchorCenters = val
		} else {
			anchorCenters = make([][]float32, height*width*_numAnchors)
			for i := 0; i < height; i++ {
				for j := 0; j < width; j++ {
					for k := 0; k < _numAnchors; k++ {
						anchorCenters[i*width*_numAnchors+j*_numAnchors+k] = []float32{float32(j * stride), float32(i * stride)}
					}
				}
			}
			//log.Println(stride, len(anchor_centers), anchor_centers)

			if len(centerCache) < 100 {
				centerCache[key] = anchorCenters
			}
		}

		// filter by det_thresh == 0.5
		var posInds []int
		for i := range scores {
			if scores[i] > detThresh {
				posInds = append(posInds, i)
			}
		}

		bboxes := distance2bbox(anchorCenters, bboxPreds)
		kpss := distance2kps(anchorCenters, kpsPreds)

		for i := range posInds {
			scoresList = append(scoresList, scores[posInds[i]])
			bboxesList = append(bboxesList, bboxes[posInds[i]])
			kpssList = append(kpssList, kpss[posInds[i]])
		}
	}

	// post process after get boxes and landmarks

	for i := range bboxesList {
		for j := 0; j < 4; j++ {
			bboxesList[i][j] /= detScale
		}
		bboxesList[i] = append(bboxesList[i], scoresList[i])

		for j := 0; j < 10; j++ {
			kpssList[i][j] /= detScale
		}
		kpssList[i] = append(kpssList[i], scoresList[i])
	}

	sort.Slice(bboxesList, func(i, j int) bool { return bboxesList[i][4] > bboxesList[j][4] })
	sort.Slice(kpssList, func(i, j int) bool { return kpssList[i][10] > kpssList[j][10] })

	keep := nms(bboxesList)

	det := make([][]float32, len(keep))
	kpss := make([][]float32, len(keep))
	for i := range keep {
		det[i] = bboxesList[keep[i]]
		kpss[i] = kpssList[keep[i]]
	}

	return det, kpss
}

func distance2bbox(points [][]float32, distance []float32) (ret [][]float32) {
	ret = make([][]float32, len(points))
	for i := range points {
		ret[i] = []float32{
			points[i][0] - distance[i*4+0],
			points[i][1] - distance[i*4+1],
			points[i][0] + distance[i*4+2],
			points[i][1] + distance[i*4+3],
		}
	}
	return
}

func distance2kps(points [][]float32, distance []float32) (ret [][]float32) {
	ret = make([][]float32, len(points))
	for i := range points {
		ret[i] = make([]float32, 10)
		for j := 0; j < 10; j = j + 2 {
			ret[i][j] = points[i][j%2] + distance[i*10+j]
			ret[i][j+1] = points[i][j%2+1] + distance[i*10+j+1]
		}
	}
	return
}

func max(a, b float32) float32 {
	if a > b {
		return a
	} else {
		return b
	}
}

func min(a, b float32) float32 {
	if a < b {
		return a
	} else {
		return b
	}
}

func nms(dets [][]float32) (ret []int) {
	if len(dets) == 0 {
		return
	}

	var order []int
	areas := make([]float32, len(dets))
	for i := range dets {
		order = append(order, i)
		areas[i] = (dets[i][2] - dets[i][0] + 1) * (dets[i][3] - dets[i][1] + 1)
	}
	for len(order) > 0 {
		i := order[0]
		ret = append(ret, i)

		var keep []int
		for j := range order[1:] {
			xx1 := max(dets[i][0], dets[order[j+1]][0])
			yy1 := max(dets[i][1], dets[order[j+1]][1])
			xx2 := min(dets[i][2], dets[order[j+1]][2])
			yy2 := min(dets[i][3], dets[order[j+1]][3])

			w := max(0.0, xx2-xx1+1)
			h := max(0.0, yy2-yy1+1)
			inter := w * h
			ovr := inter / (areas[i] + areas[order[j+1]] - inter)

			if ovr <= nmsThresh {
				keep = append(keep, order[j+1])
			}
		}

		order = keep
	}
	return
}
