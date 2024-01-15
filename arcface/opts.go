package arcface

import (
	"github.com/ivansuteja96/go-onnxruntime"
)

type ArcFaceOption func(*ArcFace) error

// Load onnx model from infightface, based on "buffalo_l" (det_10g.onnx).
func WithDetModel(onnxmodelPath string) ArcFaceOption {
	return func(h *ArcFace) (err error) {
		if h.detModel, err = onnxruntime.NewORTSession(h.ortEnv, onnxmodelPath, h.ortSessionOptions); err != nil {
			return
		}
		return
	}
}

// Load onnx model from infightface, based on "buffalo_l" (w600k_r50.onnx).
func WithArcfaceModel(onnxmodelPath string) ArcFaceOption {
	return func(h *ArcFace) (err error) {
		if h.arcfaceModel, err = onnxruntime.NewORTSession(h.ortEnv, onnxmodelPath, h.ortSessionOptions); err != nil {
			return
		}
		return
	}
}
