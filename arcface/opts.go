package arcface

import (
	"path/filepath"

	"github.com/ivansuteja96/go-onnxruntime"
)

type ArcFaceOption func(*ArcFace) error

func WithModelPath(onnxmodelPath string) ArcFaceOption {
	return func(arcFace *ArcFace) (err error) {
		arcFace.modelPath = onnxmodelPath
		return
	}
}

// Load onnx model from infightface, based on "buffalo_l" (det_10g.onnx).
func WithDetModel(onnxmodelPath string) ArcFaceOption {
	return func(arcFace *ArcFace) (err error) {
		if arcFace.detModel, err = onnxruntime.NewORTSession(arcFace.ortEnv, filepath.Join(arcFace.modelPath, onnxmodelPath), arcFace.ortSessionOptions); err != nil {
			return
		}
		return
	}
}

// Load onnx model from infightface, based on "buffalo_l" (w600k_r50.onnx).
func WithArcfaceModel(onnxmodelPath string) ArcFaceOption {
	return func(arcFace *ArcFace) (err error) {
		if arcFace.arcfaceModel, err = onnxruntime.NewORTSession(arcFace.ortEnv, filepath.Join(arcFace.modelPath, onnxmodelPath), arcFace.ortSessionOptions); err != nil {
			return
		}
		return
	}
}
