// Group Members: Evan Zhang, Alexander Garcia and Christina Monahan
// Distributed System for Tuning Hyperparameters of Neural Networks
// PAXOS election, consensus, and recovery algorithm

package main

import (
	// "encoding/json"
	"fmt"
	"io"

	// "hash/fnv"
	// "io/ioutil"
	"log"
	"os"

	// "path/filepath"
	// "plugin"
	// "sort"
	"math/rand"
	// "bufio"
	"encoding/csv"
	"strconv"
	"strings"
	"time"

	"github.com/dathoangnd/gonet"
)

// number of workers
var numWorkers int

//struct to organize data into the master function
type MasterData struct {
	id              int
	numWorkers      int
	request         []chan ModelConfig
	commands        []chan string
	dataOut         []chan [][][]float64
	models          []ModelConfig
	replies         []chan string
	corpses         chan []bool
	working         []string
	finished        []string
	toShadowMasters []chan string
	log             []string
	hb1             []chan [][]int64
	hb2             []chan [][]int64
	test            []chan [][][]float64
	training        []chan [][][]float64
}

type UIWindow struct {
	TrainData  string
	TestData   string
	ModelCount int
	Models     []ModelConfig
}

type ModelConfig struct {
	ModelID int
	Name    string
	NeuralNet
	numHiddenLayers int
	outputLayer     int
	numEpochs       int
	learningRate    float64
	momentum        float64
}

type NeuralNet struct {
	numHiddenLayers int
	outputLayer     int
	numEpochs       int
	learningRate    float64
	momentum        float64
}


func main() {
	// timer := time.Now()

	// check command line arguments
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run final.go <number of nodes>")
		return
	}

	// launches master and shadow master nodes
	launchServers(os.Args[1])

	// customize number of nodes to run a system on? on UI
	// fmt.Printf("\nRuntime: %.5f seconds\n", time.Since(timer).Seconds())
}

// Launches nodes and creates MasterData structure
func launchServers(userInput string) {
	numWorkers, _ = strconv.Atoi(userInput)

	var mrData MasterData
	mrData.numWorkers = numWorkers
	mrData.request = make([]chan ModelConfig, mrData.numWorkers) // worker <- master : for task assignment
	mrData.commands = make([]chan string, mrData.numWorkers)     // master -> worker : master sends commands
	mrData.replies = make([]chan string, mrData.numWorkers)      // master <- worker : worker reply for task completion
	mrData.corpses = make(chan []bool, mrData.numWorkers)        // master <- heartbeat : workers that have died
	mrData.working = make([]string, mrData.numWorkers)           // which tasks assigned to which workers
	mrData.training = make([]chan [][][]float64, mrData.numWorkers)      
	mrData.test = make([]chan [][][]float64, mrData.numWorkers)      

	var hb1 = make([]chan [][]int64, mrData.numWorkers+3) // heartbeat channels to neighbors for read
	var hb2 = make([]chan [][]int64, mrData.numWorkers+3) // heartbeat channels to neighbors for write
	killMaster := make(chan string, 10)                   // channel to kill Master to verify replication and recovery from log

	// initialize heartbeat tables
	for i := 0; i < mrData.numWorkers+3; i++ {
		hb1[i] = make(chan [][]int64, 1024)
		hb2[i] = make(chan [][]int64, 1024)
	}
	mrData.hb1 = hb1
	mrData.hb2 = hb2

	// initialize worker channels
	for k := 0; k < mrData.numWorkers; k++ {
		mrData.request[k] = make(chan ModelConfig)
		mrData.replies[k] = make(chan string)
		mrData.working[k] = ""
		mrData.commands[k] = make(chan string)
		mrData.training[k] = make(chan [][][]float64)
		mrData.test[k] = make(chan [][][]float64)
	}

	// initialize shadowMasters
	numShadowMasters := 2
	// shadowMaster <- master : replication
	mrData.toShadowMasters = make([]chan string, numShadowMasters)
	for j := 0; j < numShadowMasters; j++ {
		mrData.toShadowMasters[j] = make(chan string, 10)
	}

	mrData.models = make([]ModelConfig, 4)
	for i := 0; i < 4; i ++{
		var newModel ModelConfig
		mrData.models[i] = newModel
		mrData.models[i].ModelID = i
		mrData.models[i].numHiddenLayers = rand.Intn(5) 
		mrData.models[i].numEpochs = rand.Intn(20)
		mrData.models[i].learningRate = rand.Float64()
		mrData.models[i].momentum = rand.Float64()
	}


	// start nodes
	masterlog := make([]string, 0)
	var endrun = make(chan string)
	go master(mrData, hb1, hb2, masterlog, killMaster, endrun)

	go shadowMaster(mrData.toShadowMasters[0], hb1, hb2, mrData.numWorkers, mrData.numWorkers+1, mrData, killMaster)

	go shadowMaster(mrData.toShadowMasters[1], hb1, hb2, mrData.numWorkers, mrData.numWorkers+2, mrData, killMaster)


	for{
		reply := <-endrun
		fmt.Println(reply)
		if reply == "end"{
			break
		}
	}
	// wait here until "q" is entered from the command line
	// scanner := bufio.NewScanner(os.Stdin)
	// for scanner.Scan() {
	// 	text := scanner.Text()
	// 	if text == "q" {
	// 		break
	// 	}
	// }
}

// Master node
func master(mrData MasterData, hb1 []chan [][]int64, hb2 []chan [][]int64, log []string, killMaster chan string, end chan string) {
	fmt.Println("i am here")
	// initialize variables
	currentStep := "step start"
	mrData.log = log
	killHB := make(chan string, numWorkers)
	message := ""
	
	// Master Recovery: resume step after Master failure
	for i := 0; i < len(log); i++ {
		if len(log[i]) >=4 && log[i][0:4] == "step" {
			currentStep = log[i]
		}
	}
	
	// Run master heartbeat
	go masterHeartbeat(hb1, hb2, mrData.numWorkers, mrData.corpses, killHB, killMaster, mrData)

	for {
		fmt.Println("Master running: ", currentStep) // print in UI
		if currentStep == "step start" {
			// If master died partway through launching workers, find the last logged k and start from there
			k := 0
			if len(log) > 0 {
				last := strings.Split(log[len(log)-1], " ")
				if last[0] == "launch" {
					k, _ = strconv.Atoi(last[2])
				}
			}
			fmt.Println(mrData.numWorkers)
			for ; k < mrData.numWorkers; k++ {
				go worker(mrData.training[k], mrData.test[k], mrData.request[k], mrData.commands[k], mrData.replies[k], hb1, hb2, k)
				fmt.Printf("launch worker %d\n", k)
				message = fmt.Sprintf("launch worker %s", k)
				mrData.log = append(mrData.log, currentStep) //appends launch worker step to log
				mrData.toShadowMasters[0] <- message         //sends launch worker message to first Shadow Master channel
				mrData.toShadowMasters[1] <- message         //sends launch worker message to second Shadow Master channel
			}
			currentStep = "step working"
			mrData.log = append(mrData.log, currentStep) //appends load step to log
			mrData.toShadowMasters[0] <- currentStep     //sends load message to first Shadow Master channel
			mrData.toShadowMasters[1] <- currentStep     //sends load message to second Shadow Master channel

		} else if currentStep == "step working" {
			// manage the distributeTasks step
			trainpath := "datasets/mnist_test.csv"
			testpath := "datasets/mnist_train_short.csv"
			trainingdata := parseCSV(trainpath)
			testdata := parseCSV(testpath)
			fmt.Println("starting getting data")
			for k:= 0 ; k < mrData.numWorkers; k++ {
				fmt.Println("sending to worker", k)
				mrData.training[k] <- trainingdata
				mrData.test[k] <- testdata
			}
			fmt.Println("finished loading data")
			mrData = distributeTasks(mrData)
			currentStep = "step cleanup"
			mrData.log = append(mrData.log, currentStep) //appends master distributeTasks step to log
			mrData.toShadowMasters[0] <- currentStep     //sends distributeTasks message to first Shadow Master channel
			mrData.toShadowMasters[1] <- currentStep     //sends distributeTasks message to second Shadow Master channel

		} else if currentStep == "step cleanup" {
			// cleanup workers who should now be done with all tasks
			// mrData = cleanup(mrData)
			for i := 0; i < mrData.numWorkers; i++ {
				mrData.commands[i] <- "end"
			}
			currentStep = "step end"
			mrData.log = append(mrData.log, currentStep) //appends end message to log
			mrData.toShadowMasters[0] <- currentStep     //sends end message to first Shadow Master channel
			mrData.toShadowMasters[1] <- currentStep     //sends end message to second Shadow Master channel
			fmt.Println("Running master is done.")
			end <- "end"
			killHB <- "die"
			break
		}
	}
}

func distributeTasks(mrData MasterData) MasterData {
	count := 0
	loop := true

	fmt.Println("Distributing Tasks Started...")
	mrData.finished = make([]string, len(mrData.models))         // which models have completed
	for j := 0; j < len(mrData.models); j++ {
		mrData.finished[j] = "not started"
	}
	for loop {
		for i := 0; i < mrData.numWorkers; i++ {
			// checks for available workers
			if mrData.working[i] == "" {
				for j := 0; j < len(mrData.models); j++ {
					if mrData.finished[j] == "not started" {
						mrData.finished[j] = "started"
						mrData.commands[i] <- "m"
						mrData.request[i] <- (mrData.models[j])
						mrData.working[i] = strconv.Itoa(j)

						break
					}

				}
			}
			// checks for replies and dead workers
			select {
			case message := <-mrData.replies[i]:
				replied := strings.Split(message, "_")
				workerID, _ := strconv.Atoi(replied[1])
				mrData.working[workerID] = ""
				modelID, _ := strconv.Atoi(replied[0])
				mrData.finished[modelID] = "finished"
				count++
			case coffins := <-mrData.corpses:
				for j := 0; j < mrData.numWorkers; j++ {
					if coffins[j] == true {
						mrData.hb1[j] = make(chan [][]int64, numWorkers+3)
						mrData.hb2[j] = make(chan [][]int64, numWorkers+3)
						mrData.request[j] = make(chan ModelConfig)
						mrData.replies[j] = make(chan string)
						go worker(mrData.training[j], mrData.test[j], mrData.request[j], mrData.commands[j], mrData.replies[j], mrData.hb1, mrData.hb2, j)
						tempModelID, _ := strconv.Atoi(mrData.working[j])
						mrData.finished[tempModelID] = "not started"
						mrData.working[j] = ""
						coffins[j] = false
					}
				}
			default:
			}
		}

		// checks that all models have completed
		if count >= len(mrData.models) {
			check := true

			for a := 0; a < mrData.numWorkers; a++ {
				if mrData.working[a] != "" {
					check = false
				}
			}
			if check == true {
				loop = false
			}
		}
	}
	fmt.Println("\nDistributing Finished")
	return mrData
}

func worker(train chan [][][]float64, test chan [][][]float64, frommaster chan ModelConfig, commands chan string, reply chan string, hb1 []chan [][]int64, hb2 []chan [][]int64, k int) {
	var endHB = make(chan string)
	go heartbeat(hb1, hb2, k, endHB)
	task := ""
	var trainingdata [][][]float64
	var testdata [][][]float64
	var indivModel ModelConfig
	for {
		// read task from channel
		trainingdata =  <-train
		testdata = <-test
		task = <-commands
		fmt.Println(task)
		fmt.Println("recieved command")
		indivModel = <-frommaster
		tasks := strings.Split(task, "_")
		if tasks[0] == "end" {
			
			fmt.Println("ending worker", k)
			endHB <- "end"
			return
		}
		if tasks[0] == "m" {
			fmt.Print("started task")
			runNeuralNetwork(indivModel, trainingdata, testdata)
		}
		reply <- strconv.Itoa(k) + "_" +strconv.Itoa(indivModel.ModelID)
	}
}

// Shuts down worker nodes
func cleanup(mrData MasterData) MasterData {
	// Check that all workers shutdown
	for i := 0; i < mrData.numWorkers; i++ {
		mrData.commands[i] <- "end_0_" + strconv.Itoa(i)
		msg := <-mrData.replies[i]
		num, _ := strconv.Atoi(msg)
		mrData.working[num] = ""
	}
	return mrData
}

func runNeuralNetwork(model ModelConfig, train [][][]float64, test [][][]float64) {

	// train := parseCSV("../datasets/mnist_train_short.csv")
	// test := parseCSV("../datasets/mnist_train_short.csv")
	// train := parseCSV("../datasets/exams.csv")
	// test := parseCSV("../datasets/exam.csv")

	// Create a neural network
	// 2 nodes in the input layer
	// 2 hidden layers with 4 nodes each
	// 1 node in the output layer
	// The problem is classification, not regression
	
	fmt.Println("running network", model.ModelID)
	nn := gonet.New(len(train[0][0]), []int{100, 32}, 10, true)

	// Train the network
	// Run for 3000 epochs
	// The learning rate is 0.4 and the momentum factor is 0.2
	// Enable debug mode to log learning error every 1000 iterations
	nn.Train(train, model.numEpochs, model.learningRate, model.momentum, true)

	// Predict
	totalcorrect := 0.0
	for i := 0; i < len(test); i++ {
		// fmt.Print(MinMax(test[i][1]), " ", MinMax(nn.Predict(test[i][0])), " | ")
		// if i%15 == 0{
		// 	fmt.Println()
		// }

		if MinMax(test[i][1]) == MinMax(nn.Predict(test[i][0])) {
			totalcorrect += 1.0
		}
	}
	printModelParam(model)
	fmt.Printf("Model %d : Percent correct: %.2f percent\n", model.ModelID, totalcorrect/float64(len(test)) * 100.0)

	// // Save the model
	// nn.Save("model.json")

	// // Load the model
	// nn2, err := gonet.Load("model.json")
	// if err != nil {
	// 	log.Fatal("Load model failed.")
	// }

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

// Helper function to find max
func max(x, y int64) int64 {
	if x > y {
		return x
	}
	return y
}

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

func printModelParam(model ModelConfig){
	fmt.Println("ID", model.ModelID, ", epochs", model.numEpochs, ", learning rate", model.learningRate,
				", momentum", model.momentum, ", hidden layers", model.numHiddenLayers)
}

// Update heartbeat tables for Master, 2 ShadowMasters and 8 workers
func updateTable(index int, hbtable [][]int64, counter int, hb1 []chan [][]int64, hb2 []chan [][]int64) [][]int64 {
	next := index + 1
	prev := index - 1
	if prev < 0 {
		prev = numWorkers + 2
	}
	if next > numWorkers+2 {
		next = 0
	}

	temp := make([][]int64, numWorkers+3)
	neighbor1 := make([][]int64, numWorkers+3)
	neighbor2 := make([][]int64, numWorkers+3)

	for i := 0; i < numWorkers+3; i++ {
		neighbor1[i] = make([]int64, 2)
		neighbor1[i][0] = hbtable[i][0]
		neighbor1[i][1] = hbtable[i][1]
		neighbor2[i] = make([]int64, 2)
		neighbor2[i][0] = hbtable[i][0]
		neighbor2[i][1] = hbtable[i][1]
		temp[i] = make([]int64, 2)
		temp[i][0] = hbtable[i][0]
		temp[i][1] = hbtable[i][1]
	}
	loop2 := true
	if counter%1 == 0 {
		for loop2 {
			select {
			case neighbor1_1 := <-hb1[next]:
				neighbor1 = neighbor1_1
			default:
				loop2 = false
			}
		}
		for i := 0; i < numWorkers+3; i++ {
			temp[i][0] = max(neighbor1[i][0], hbtable[i][0])
			temp[i][1] = max(neighbor1[i][1], hbtable[i][1])
		}
		loop2 = true

		for loop2 {
			select {
			case neighbor2_1 := <-hb2[prev]:
				neighbor2 = neighbor2_1
			default:
				loop2 = false
			}
		}
		for i := 0; i < numWorkers+3; i++ {
			temp[i][0] = max(neighbor2[i][0], hbtable[i][0])
			temp[i][1] = max(neighbor2[i][1], hbtable[i][1])
		}
	}
	now := time.Now().Unix() // current local time
	temp[index][0] = hbtable[index][0] + 1
	temp[index][1] = now
	// send table
	hb1[index] <- temp
	hb2[index] <- temp
	return temp
}

// Heartbeat function for all workers
func heartbeat(hb1 []chan [][]int64, hb2 []chan [][]int64, k int, endHB chan string) {
	now := time.Now().Unix() // current local time
	counter := 0
	hbtable := make([][]int64, numWorkers+3)
	// initialize hbtable
	for i := 0; i < numWorkers+3; i++ {
		hbtable[i] = make([]int64, 2)
		hbtable[i][0] = 0
		hbtable[i][1] = now
	}

	for {
		fmt.Print("w hb ")
		time.Sleep(100 * time.Millisecond)
		hbtable = updateTable(k, hbtable, counter, hb1, hb2)
		counter++
		select {
		case reply := <-endHB:
			if reply == "end" {
				return
			}
		default:
		}
	}
}

// Heartbeat function for Master
func masterHeartbeat(hb1 []chan [][]int64, hb2 []chan [][]int64, k int, corpses chan []bool, kill chan string, killMaster chan string, mrData MasterData) {
	now := time.Now().Unix() // current local time
	counter := 0
	currentTable := make([][]int64, numWorkers+3)
	previousTable := make([][]int64, numWorkers+3)

	fmt.Println("master heartbeat started")

	// initialize hbtable
	for i := 0; i < numWorkers+3; i++ {
		currentTable[i] = make([]int64, 2)
		previousTable[i] = make([]int64, 2)
		currentTable[i][0] = 0
		previousTable[i][0] = 0
		currentTable[i][1] = now
		previousTable[i][1] = now
	}
	deadWorkers := make([]bool, numWorkers+3)
	for i := 0; i < numWorkers+3; i++ {
		deadWorkers[i] = false
	}
	for {
		select {
		case reply := <-kill:
			if reply == "die" {
				return
			}
		default:
		}
		time.Sleep(100 * time.Millisecond)
		currentTable = updateTable(k, previousTable, counter, hb1, hb2)
		for i := 0; i < numWorkers+3; i++ {
			if currentTable[k][1]-previousTable[i][1] > 2 {
				fmt.Println(currentTable[k][1], previousTable[i][1])
				if i == numWorkers+1 || i == numWorkers+2 {
					fmt.Println("Shadow master died")
					go shadowMaster(mrData.toShadowMasters[i-numWorkers], mrData.hb1, mrData.hb2, mrData.numWorkers, i, mrData, killMaster)
				} else {
					fmt.Println("\n\n-------------------killed worker :", i, "\n")
					deadWorkers[i] = true
				}
			}
		}
		previousTable = currentTable
		corpses <- deadWorkers
		counter++
	}
	fmt.Println("master heartbeat ended")
}

// Shadow Master Node
func shadowMaster(copier chan string, hb1 []chan [][]int64, hb2 []chan [][]int64, masterID int, selfID int, mrData MasterData, kill chan string) {
	// replicate logs to two shadowmasters that monitor if the master dies
	logs := make([]string, 0)
	killHB := make(chan string, 3)
	var isMasterDead = make(chan bool, 1)
	go shadowHeartbeat(hb1, hb2, masterID, isMasterDead, selfID, killHB)
	masterNotDead := true

	for masterNotDead {
		select {
		case copy := <-copier:
			fmt.Println("shadow", selfID, copy)
			logs = append(logs, copy)
			// check if to die
			currentStep := copy
			if currentStep == "step start" {
				// update logs
				currentStep = "step load"
				mrData.log = append(mrData.log, currentStep)

			}else if currentStep == "step working" {
				// master working
				currentStep = "step cleanup"
				mrData.log = append(mrData.log, currentStep)

			} else if currentStep == "step cleanup" {
				// cleanup
				currentStep = "step end"
				mrData.log = append(mrData.log, currentStep)
				killHB <- "kill"
				return
			}
		case isDead := <-isMasterDead:
			masterNotDead = isDead
		default:
		}
		if !masterNotDead {
			fmt.Println("Shadow master becomes running master.")
			var endrun = make(chan string)
			go master(mrData, hb1, hb2, logs, kill, endrun)
			masterNotDead = true
		}
	}
}

// Heartbeat for Shadow Master
func shadowHeartbeat(hb1 []chan [][]int64, hb2 []chan [][]int64, masterID int, isMasterAlive chan bool, selfID int, killHB chan string) string {
	now := time.Now().Unix() // current local time
	counter := 0
	currentTable := make([][]int64, numWorkers+3)
	previousTable := make([][]int64, numWorkers+3)

	// initialize hbtable
	for i := 0; i < numWorkers+3; i++ {
		currentTable[i] = make([]int64, 2)
		previousTable[i] = make([]int64, 2)
		currentTable[i][0] = 0
		previousTable[i][0] = 0
		currentTable[i][1] = now
		previousTable[i][1] = now
	}

	for {
		select {
		case reply := <-killHB:
			return reply
		default:
		}
		time.Sleep(100 * time.Millisecond)
		currentTable = updateTable(selfID, previousTable, counter, hb1, hb2)

		if currentTable[selfID][1]-previousTable[masterID][1] > 2{
			if selfID == masterID+1 {
				fmt.Println("\n----- The Running Master has died -----\n")
				isMasterAlive <- false
				currentTable = updateTable(masterID, currentTable, counter, hb1, hb2)
			}
		}
		previousTable = currentTable
		counter++
	}

}
