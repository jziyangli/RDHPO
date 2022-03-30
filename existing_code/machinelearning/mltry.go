package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"

	// "time"
	"github.com/dathoangnd/gonet"
	// "github.com/cdipaolo/goml/cluster"
	// "github.com/cdipaolo/goml/linear"
)

//TODO: parse data
//TODO: parse hyperparameters

func parseCSV(path string) [][][]float64 {
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
		if index != 0 && len(record) > 1 {
			floatarr := make([]float64, len(record)-1)
			expected := make([]float64, 10)
			for i := 0; i < len(record); i++ {
				if s, err := strconv.ParseFloat(record[i], 64); err == nil {
					// fmt.Println(s)
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
			// fmt.Println(expected)
			one_entry := [][]float64{floatarr, expected}

			data = append(data, one_entry)
			// fmt.Println(one_entry)

		}
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		index++
	}
	// fmt.Println(data)
	return data
}

// index 0 [[28x28], expected_value]
// index 1

func main() {

	train := parseCSV("../datasets/mnist_train_short.csv")
	test := parseCSV("../datasets/mnist_train_short.csv")
	// train := parseCSV("../datasets/exams.csv")
	// test := parseCSV("../datasets/exam.csv")
	// XOR traning data
	// trainingData := [][][]float64{
	// 	{{0, 0}, {0}},
	// 	{{0, 1}, {1}},
	// 	{{1, 0}, {1}},
	// 	{{1, 1}, {0}},
	// }
	// for i := 0 ; i < 10; i++{
	// 	fmt.Println(test[i][1], "\n\n")

	// }

	// Create a neural network
	// 2 nodes in the input layer
	// 2 hidden layers with 4 nodes each
	// 1 node in the output layer
	// The problem is classification, not regression
	
	// fmt.Println("size of input", len(train[0][0]))
	nn := gonet.New(len(train[0][0]), []int{100, 32}, 10, true)
	// func New(nInputs int, nHiddens []int, nOutputs int, isRegression bool) NN
	// Train the network
	// Run for 3000 epochs
	// The learning rate is 0.4 and the momentum factor is 0.2
	// Enable debug mode to log learning error every 1000 iterations
	nn.Train(train, 30, 0.2, 0.2, true)

	// Predict
	totalcorrect := 0.0
	for i := 0; i < len(test); i++ {
		fmt.Print(MinMax(test[i][1]), " ", MinMax(nn.Predict(test[i][0])), " | ")
		if i%15 == 0{
			fmt.Println()
		}
		// fmt.Println("predicted", MinMax(nn.Predict(test[i][0])))
		// fmt.Printf("actual: %d, predicted: %d\n", MinMax(test[i][1][0])[1], MinMax(nn.Predict(test[i][0]))[1])

		if MinMax(test[i][1]) == MinMax(nn.Predict(test[i][0])) {
			totalcorrect += 1.0
		}
	}
	fmt.Printf("Percent correct: %.2f percent\n", totalcorrect/float64(len(test)) * 100.0)



	// Save the model
	// nn.Save("model.json")

	// // Load the model
	// nn2, err := gonet.Load("model.json")
	// if err != nil {
	// 	log.Fatal("Load model failed.")
	// }
	// fmt.Printf("%f XOR %f => %f\n", testInput[0], testInput[1], nn2.Predict(testInput)[0])
	// 1.000000 XOR 0.000000 => 0.943074

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
