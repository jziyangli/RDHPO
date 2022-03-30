package mlfunctions

import (
	"fmt"
	"log"
	"io"
	"os"
	"encoding/csv"
	"strconv"
	"time"
	"github.com/dathoangnd/gonet"
)

func parseCSV(path string) [][][]float64{
	data := make([][][]float64, 0)

	// Open the file
	csvfile, err := os.Open(path)
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}

	r := csv.NewReader(csvfile)
	index := 0

	// Parse Data
	for {
		// Read each record from csv
		record, err := r.Read()
		if index != 0{
			floatarr := make([]float64, len(record) + 1)
			expected := make([]float64, 1)
			for i := 0; i < len(record); i++ {
				if s, err := strconv.ParseFloat(record[i], 64); err == nil {
					if i == 0{
						expected[0] = s
					}else{
						floatarr[i] = s
					}
					// fmt.Println(s)
				}
			}
			if len(floatarr[1:]) == 0{
				break
			}
			one_entry := [][]float64{floatarr[1:], expected}
			
			data = append(data, one_entry)
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatal(err)
			}
			// fmt.Println(index)
		}
		index++
	}
	// fmt.Println(data)
	return data
}

func runNeuralNet(training [][][]float64, test [][][]float64, inputNodes int, hiddenLayers []int, outputLayer int, numEpochs int, learningRate float64, momentum float64){
	// Create a neural network
	// 2 nodes in the input layer
	// 2 hidden layers with 4 nodes each []int{4, 4}
	// 1 node in the output layer
	// The problem is classification, not regression
	nn := gonet.New(inputNodes, hiddenLayers, outputLayer, false)

	// Train the network
	// Run for 3000 epochs
	// The learning rate is 0.4 and the momentum factor is 0.2
	// Enable debug mode to log learning error every 1000 iterations
	nn.Train(training, numEpochs, learningRate, momentum, true)

	// Predict
	// testInput := []float64{1, 0}
	// fmt.Printf("test input: %f  %f => %f\n", testInput[0], testInput[1], nn.Predict(testInput)[0])
	// // Save the model
	// nn.Save("model.json")
	// // Load the model
	// nn2, err := gonet.Load("model.json")
	// if err != nil {
	// 	log.Fatal("Load model failed.")
	// }
	// fmt.Printf("%f XOR %f => %f\n", testInput[0], testInput[1], nn2.Predict(testInput)[0])
	// // 1.000000 XOR 0.000000 => 0.943074
}

