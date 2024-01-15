package main

import (
	"log"

	"github.com/disintegration/imaging"
	"github.com/jack139/arcface-go/arcface"
)

const (
	testImagePath = "diego.jpg"
)

func main() {
	// by default "./models/det_10g.onnx" and "./models/w600k_r50.onnx"
	arcFace, err := arcface.New(
	// arcface.WithDetModel("./models/det_10g.onnx"),
	// arcface.WithArcfaceModel("./models/w600k_r50.onnx"),
	)
	if err != nil {
		log.Fatal("New() error: ", err)
	}

	// load image
	srcImage, err := imaging.Open(testImagePath)
	if err != nil {
		log.Fatal("Open image error: ", err)
	}

	dets, kpss, err := arcFace.FaceDetect(srcImage)
	if err != nil {
		log.Fatal("FaceDetect() error: ", err)
	}

	log.Println("face num: ", len(kpss))

	if len(dets) == 0 {
		log.Println("No face detected.")
		return
	}

	/*
		// crop face by detect boxes without normalization
		sr := image.Rectangle{
			image.Point{int(dets[0][0]), int(dets[0][1])},
			image.Point{int(dets[0][2]), int(dets[0][3])},
		}
		src2 := imaging.Crop(srcImage, sr)
		_ = imaging.Save(src2, "crop_face.jpg")
	*/

	// just use the first face data, which score is the highest
	features, normFace, err := arcFace.FaceFeatures(srcImage, kpss[0])
	if err != nil {
		log.Fatal("FaceFeatures() error: ", err)
	}
	log.Println("features: ", features)

	// normalized face image
	if err = imaging.Save(normFace, "norm_face.jpg"); err != nil {
		log.Fatal("Save image error: ", err)
	}
}
