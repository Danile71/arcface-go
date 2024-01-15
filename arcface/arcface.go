// Package to provide face-detection and features-retrieving
package arcface

import (
	"errors"
	"image"

	"github.com/ivansuteja96/go-onnxruntime"
	//"github.com/disintegration/imaging"
)

const (
	detModelInputSize  = 224
	faceAlignImageSize = 112
)

type ArcFace struct {
	detModel     *onnxruntime.ORTSession
	arcfaceModel *onnxruntime.ORTSession

	ortEnv            *onnxruntime.ORTEnv
	ortSessionOptions *onnxruntime.ORTSessionOptions
}

// Load onnx model from infightface, based on "buffalo_l" (det_10g.onnx, w600k_r50.onnx).
// onnxmodel_path is the path way to onnx models.
func New(opts ...ArcFaceOption) (arcFace *ArcFace, err error) {
	arcFace = &ArcFace{
		ortEnv:            onnxruntime.NewORTEnv(onnxruntime.ORT_LOGGING_LEVEL_ERROR, "development"),
		ortSessionOptions: onnxruntime.NewORTSessionOptions(),
	}

	for _, opt := range opts {
		if err = opt(arcFace); err != nil {
			return
		}
	}

	return
}

// Detect face in src image,
// return face boxes and landmarks, ordered by predict scores.
func (arcFace *ArcFace) FaceDetect(src image.Image) ([][]float32, [][]float32, error) {
	shape1 := []int64{1, 3, detModelInputSize, detModelInputSize}
	input1, detScale := preprocessImage(src, detModelInputSize)

	if arcFace.detModel == nil {
		if err := WithDetModel("./models/det_10g.onnx")(arcFace); err != nil {
			return nil, nil, err
		}
	}

	// face detect model inference
	res, err := arcFace.detModel.Predict([]onnxruntime.TensorValue{
		{
			Value: input1,
			Shape: shape1,
		},
	})
	if err != nil {
		return nil, nil, err
	}

	if len(res) == 0 {
		return nil, nil, errors.New("Fail to get result")
	}

	dets, kpss := processResult(res, detScale)

	return dets, kpss, nil
}

// Get face features by Arcface
// Parameter src is original image, lmk is face landmark detected by FaceDetect(),
// return features in a arrary, and norm_crop image
func (arcFace *ArcFace) FaceFeatures(src image.Image, lmk []float32) ([]float32, image.Image, error) {
	aimg, err := normCrop(src, lmk)
	if err != nil {
		return nil, nil, err
	}

	// save normalization crop face for test
	//_ = imaging.Save(aimg, "data/crop_norm.jpg")

	// prepare input data
	shape2 := []int64{1, 3, faceAlignImageSize, faceAlignImageSize}
	input2 := preprocessFace(aimg, faceAlignImageSize)

	if arcFace.arcfaceModel == nil {
		if err = WithArcfaceModel("./models/w600k_r50.onnx")(arcFace); err != nil {
			return nil, nil, err
		}
	}

	// face features modle inference
	res2, err := arcFace.arcfaceModel.Predict([]onnxruntime.TensorValue{
		{
			Value: input2,
			Shape: shape2,
		},
	})
	if err != nil {
		return nil, nil, err
	}

	if len(res2) == 0 {
		return nil, nil, errors.New("Fail to get result")
	}

	return res2[0].Value.([]float32), aimg, nil
}
