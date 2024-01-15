package arcface

import (
	"fmt"
	"image"

	"github.com/jack139/arcface-go/gocvx"
)

var (
	// arcface matrix from insightface/utils/face_align.py
	arcfaceSrc = []gocvx.Point2f{
		{X: 38.2946, Y: 51.6963},
		{X: 73.5318, Y: 51.5014},
		{X: 56.0252, Y: 71.7366},
		{X: 41.5493, Y: 92.3655},
		{X: 70.7299, Y: 92.2041},
	}
)

// Crop face image and normalization
func normCrop(srcImage image.Image, lmk []float32) (image.Image, error) {
	// similarity transform
	m := estimateNorm(lmk)
	defer m.Close()

	// print out the 2×3 transformation matrix
	//printM(m)

	// transfer to Mat
	src, err := gocvx.ImageToMatRGB(srcImage)
	if err != nil {
		return nil, err
	}
	defer src.Close()

	dst := src.Clone()
	defer dst.Close()

	// affine transformation to an image (Mat)
	gocvx.WarpAffine(src, &dst, m, image.Point{faceAlignImageSize, faceAlignImageSize})

	// Mat transfer to image
	aimg, err := dst.ToImage()
	if err != nil {
		return nil, err
	}

	return aimg, nil
}

// equal to python: skimage.transform.SimilarityTransform()
func estimateNorm(lmk []float32) gocvx.Mat {
	dst := make([]gocvx.Point2f, 5)
	for i := 0; i < 5; i++ {
		dst[i] = gocvx.Point2f{X: lmk[i*2], Y: lmk[i*2+1]}
	}

	pvsrc := gocvx.NewPoint2fVectorFromPoints(arcfaceSrc)
	defer pvsrc.Close()

	pvdst := gocvx.NewPoint2fVectorFromPoints(dst)
	defer pvdst.Close()

	inliers := gocvx.NewMat()
	defer inliers.Close()
	method := 4 // cv2.LMEDS
	ransacProjThreshold := 3.0
	maxiters := uint(2000)
	confidence := 0.99
	refineIters := uint(10)

	m := gocvx.EstimateAffinePartial2DWithParams(pvdst, pvsrc, inliers, method,
		ransacProjThreshold, maxiters, confidence, refineIters)

	return m
}

// print matrix, for test
func printM(m gocvx.Mat) {
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			fmt.Printf("%v ", m.GetDoubleAt(i, j))
		}
		fmt.Printf("\n")
	}
}
