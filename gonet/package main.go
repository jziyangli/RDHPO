package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"

	"github.com/dathoangnd/gonet"
)

type Config struct {
	NumHiddenLayers int
	NumEpochs       int
	LearningRate    float64
	Momentum        float64
	ModelID         string
}

type Main struct {
	conf     chan Config
	training chan [][][]float64
	testing  chan [][][]float64
	run      chan bool
	out      chan string
}

func main() {
	const numWorkers int = 1
	train := parseCSV("data/mnist_train.csv")
	test := parseCSV("data/mnist_test.csv")

	var config Config
	config.NumHiddenLayers = 3
	config.NumEpochs = 25
	config.LearningRate = 0.01
	config.Momentum = 0.9
	config.ModelID = "default"

	var workers [numWorkers]Main

	out := make(chan string)
	//runGonet(config, train, test)
	for i := 0; i < numWorkers; i++ {
		var mainStruct Main
		mainStruct.conf = make(chan Config)
		mainStruct.training = make(chan [][][]float64)
		mainStruct.testing = make(chan [][][]float64)
		mainStruct.run = make(chan bool)
		mainStruct.out = out
		workers[i] = mainStruct
	}
	//conf2 := make(chan Config)
	//training2 := make(chan [][][]float64)
	//testing2 := make(chan [][][]float64)
	//run2 := make(chan bool)
	for i := 0; i < numWorkers; i++ {
		go distribute(workers[i].conf, workers[i].training, workers[i].testing, workers[i].run, workers[i].out)
	}
	//go distribute(conf2, training2, testing2, run2, out2)
	for i := 0; i < numWorkers; i++ {
		workers[i].conf <- config
		workers[i].training <- train
		workers[i].testing <- test
		workers[i].run <- true
	}
	//conf2 <- config
	//training2 <- train
	//testing2 <- test
	//run2 <- true
	for i := 0; i < numWorkers; i++ {
		retval := <-out
		fmt.Println(retval)
	}
}

func distribute(config chan Config, train chan [][][]float64, test chan [][][]float64, run chan bool, out chan string) {
	var trainingdata [][][]float64
	var testdata [][][]float64
	var conf Config
	for i := 0; i < 3; i++ {
		select {
		case trainingdata = <-train:
		case testdata = <-test:
		case conf = <-config:
		}
	}
	r := <-run
	if r {
		out <- runGonet(conf, trainingdata, testdata)
	}
}

func runGonet(m Config, train [][][]float64, test [][][]float64) string {
	fmt.Println("Starting Gonet...")
	hidden := make([]int, m.NumHiddenLayers)
	for j := 0; j < m.NumHiddenLayers; j++ {
		hidden[j] = 32
	}
	nn := gonet.New(len(train[0][0]), hidden, len(train[0][1]), true)

	// Train the network, use epochs, learning rate, and momentum factor from UI
	timer := time.Now()
	nn.Train(train, m.NumEpochs, m.LearningRate, m.Momentum, true)
	s := fmt.Sprintf("%s Results - ", m.ModelID)
	runtime := fmt.Sprintf("Runtime: %.5f s, ", time.Since(timer).Seconds())

	// Predict
	totalcorrect := 0.0
	for i := 0; i < len(test); i++ {
		if MinMax(test[i][1]) == MinMax(nn.Predict(test[i][0])) {
			totalcorrect += 1.0
		}
	}

	acc := fmt.Sprintf("Accuracy: %.2f%%", totalcorrect/float64(len(test))*100.0)
	s = s + runtime + acc

	fmt.Println("Finished Gonet")
	return s
}

func parseCSV(path string) [][][]float64 {
	data := make([][][]float64, 0)
	x := make([][][]float64, 0)

	// Open the file
	csvfile, err := os.Open(path)
	if err != nil {
		// log.Fatalln("Couldn't open the csv file", err)
		fmt.Println("Could not open csv:", path)
		return x
	}

	r := csv.NewReader(csvfile)

	index := 0

	// Parse Data
	for {
		// Read each record from csv
		record, err := r.Read()
		if index != 0 && len(record) > 1 {
			floatarr := make([]float64, len(record)-1)
			expected := make([]float64, 10)
			for i := 0; i < len(record); i++ {
				if s, err := strconv.ParseFloat(record[i], 64); err == nil {
					if i == 0 {
						expected[int(s)] = 1
					} else {
						floatarr[i-1] = s
					}
				}
			}
			if len(floatarr) == 0 {
				break
			}
			oneEntry := [][]float64{floatarr, expected}
			data = append(data, oneEntry)
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Println("CSV could not be read")
			return x
			// log.Fatal(err)
		}
		index++
	}
	return data
}

func MinMax(array []float64) int {
	index := 0
	max := 0.0
	for i, value := range array {
		if max < value {
			index = i
			max = value
		}
	}
	return index
}
