package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/andlabs/ui"
	_ "github.com/andlabs/ui/winmanifest"
	"github.com/dathoangnd/gonet"
)

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
	Model2Params
	Model3Params
}

type NeuralNet struct {
	InputNodes      int
	NumHiddenLayers int
	OutputNodes     int
	NumEpochs       int
	LearningRate    float64
	Momentum        float64
}

type Model2Params struct {
	Layers   int
	Learning string
}

type Model3Params struct {
	Trees    int
	MaxDepth int
}

var mainwin *ui.Window
var modelCount int = 0
var models []ui.Control = make([]ui.Control, 5)
var windowData UIWindow

func makeModelParam(m ModelConfig) ui.Control {
	hbox := ui.NewHorizontalBox()
	hbox.SetPadded(true)

	grid := ui.NewGrid()
	grid.SetPadded(true)

	hbox.Append(grid, false)

	form := ui.NewForm()
	form.SetPadded(true)
	grid.Append(form,
		0, 0, 1, 1,
		false, ui.AlignFill, false, ui.AlignFill)

	initial := 0
	if m.Name == "Model 2" {
		initial = 1
	}
	if m.Name == "Model 3" {
		initial = 2
	}
	alignment := ui.NewCombobox()
	// note that the items match with the values of the uiDrawTextAlign values
	alignment.Append("Neural Network")
	alignment.Append("Model 2")
	alignment.Append("Model 3")
	alignment.SetSelected(initial)
	alignment.OnSelected(func(*ui.Combobox) {
		s := alignment.Selected()
		if s == 0 {
			windowData.Models[m.ModelID].Name = "Neural Network"
			hbox1 := ui.NewHorizontalBox()
			hbox1.SetPadded(true)
			grid.Append(hbox1,
				0, 1, 1, 1,
				false, ui.AlignFill, false, ui.AlignFill)

			// InputNodes   	int
			// NumHiddenLayers 	int
			// OutputNodes		int
			// NumEpochs		int
			// LearningRate	float64
			// Momentum		float64

			// # of input nodes
			inputNodes := ui.NewSpinbox(0, 10000)
			inputNodes.OnChanged(func(*ui.Spinbox) {
				windowData.Models[m.ModelID].InputNodes = inputNodes.Value()
			})

			form1 := ui.NewForm()
			form1.SetPadded(true)
			hbox1.Append(form1, false)
			form1.Append("# Input Nodes", inputNodes, false)

			// # of hidden layers
			layers := ui.NewSpinbox(0, 10000)
			layers.OnChanged(func(*ui.Spinbox) {
				windowData.Models[m.ModelID].NumHiddenLayers = layers.Value()
			})

			form2 := ui.NewForm()
			form2.SetPadded(true)
			hbox1.Append(form2, false)
			form2.Append("# Hidden Layers", layers, false)

			// # of output nodes
			outputNodes := ui.NewSpinbox(0, 10000)
			outputNodes.OnChanged(func(*ui.Spinbox) {
				windowData.Models[m.ModelID].OutputNodes = outputNodes.Value()
			})

			form3 := ui.NewForm()
			form3.SetPadded(true)
			hbox1.Append(form3, false)
			form3.Append("# Output Nodes", outputNodes, false)

			// # of epochs
			epochs := ui.NewSpinbox(0, 10000)
			epochs.OnChanged(func(*ui.Spinbox) {
				windowData.Models[m.ModelID].NumEpochs = epochs.Value()
			})

			form4 := ui.NewForm()
			form4.SetPadded(true)
			hbox1.Append(form4, false)
			form4.Append("# of Epochs", epochs, false)

			// learning rate
			lrate := ui.NewEntry()
			windowData.Models[m.ModelID].LearningRate = 0.0
			lrate.OnChanged(func(*ui.Entry) {
				f, err := strconv.ParseFloat(lrate.Text(), 64)
				if err == nil {
					windowData.Models[m.ModelID].LearningRate = f
				} else {
					ui.MsgBoxError(mainwin,
						"Not a Number.",
						"Please enter a numerical value.")
					lrate.SetText("")
				}
			})

			form5 := ui.NewForm()
			form5.SetPadded(true)
			hbox1.Append(form5, false)
			form5.Append("Learning Rate", lrate, false)

			// momentum
			momentum := ui.NewEntry()
			windowData.Models[m.ModelID].Momentum = 0.0
			momentum.OnChanged(func(*ui.Entry) {
				f, err := strconv.ParseFloat(momentum.Text(), 64)
				if err == nil {
					windowData.Models[m.ModelID].Momentum = f
				} else {
					ui.MsgBoxError(mainwin,
						"Not a Number.",
						"Please enter a numerical value.")
					momentum.SetText("")
				}
			})

			form6 := ui.NewForm()
			form6.SetPadded(true)
			hbox1.Append(form6, false)
			form6.Append("Momentum", momentum, false)

			// end

		} else if s == 1 {
			windowData.Models[m.ModelID].Name = "Model 2"
			hbox1 := ui.NewHorizontalBox()
			hbox1.SetPadded(true)
			grid.Append(hbox1,
				0, 1, 1, 1,
				false, ui.AlignFill, false, ui.AlignFill)
			// layers and learning rate

			layers := ui.NewSpinbox(0, 10000)
			layers.OnChanged(func(*ui.Spinbox) {
				windowData.Models[m.ModelID].Layers = layers.Value()
			})

			form2 := ui.NewForm()
			form2.SetPadded(true)
			hbox1.Append(form2, false)
			form2.Append("numLayers", layers, false)

			lrate := ui.NewEntry()
			lrate.SetText("0.1")
			windowData.Models[m.ModelID].Learning = "0.1"
			lrate.OnChanged(func(*ui.Entry) {
				windowData.Models[m.ModelID].Learning = lrate.Text()
			})

			form1 := ui.NewForm()
			form1.SetPadded(true)
			hbox1.Append(form1, false)
			form1.Append("learning rate", lrate, false)
		} else {
			windowData.Models[m.ModelID].Name = "Model 3"
			hbox1 := ui.NewHorizontalBox()
			hbox1.SetPadded(true)
			grid.Append(hbox1,
				0, 1, 1, 1,
				false, ui.AlignFill, false, ui.AlignFill)
			// numTrees and max depth
			numTrees := ui.NewSpinbox(0, 10000)
			numTrees.OnChanged(func(*ui.Spinbox) {
				windowData.Models[m.ModelID].Trees = numTrees.Value()
			})

			form1 := ui.NewForm()
			form1.SetPadded(true)
			hbox1.Append(form1, false)
			form1.Append("numTrees", numTrees, false)

			maxDepth := ui.NewSpinbox(0, 10000)
			maxDepth.OnChanged(func(*ui.Spinbox) {
				windowData.Models[m.ModelID].MaxDepth = maxDepth.Value()
			})

			form2 := ui.NewForm()
			form2.SetPadded(true)
			hbox1.Append(form2, false)
			form2.Append("maxDepth", maxDepth, false)
		}
	})

	if m.Name == "Neural Network" {
		hbox1 := ui.NewHorizontalBox()
		hbox1.SetPadded(true)
		grid.Append(hbox1,
			0, 1, 1, 1,
			false, ui.AlignFill, false, ui.AlignFill)

		// # of input nodes
		inputNodes := ui.NewSpinbox(0, 10000)
		inputNodes.SetValue(windowData.Models[m.ModelID].InputNodes)
		inputNodes.OnChanged(func(*ui.Spinbox) {
			windowData.Models[m.ModelID].InputNodes = inputNodes.Value()
		})

		form1 := ui.NewForm()
		form1.SetPadded(true)
		hbox1.Append(form1, false)
		form1.Append("# Input Nodes", inputNodes, false)

		// # of hidden layers
		layers := ui.NewSpinbox(0, 10000)
		layers.SetValue(windowData.Models[m.ModelID].NumHiddenLayers)
		layers.OnChanged(func(*ui.Spinbox) {
			windowData.Models[m.ModelID].NumHiddenLayers = layers.Value()
		})

		form2 := ui.NewForm()
		form2.SetPadded(true)
		hbox1.Append(form2, false)
		form2.Append("# Hidden Layers", layers, false)

		// # of output nodes
		outputNodes := ui.NewSpinbox(0, 10000)
		outputNodes.SetValue(windowData.Models[m.ModelID].OutputNodes)
		outputNodes.OnChanged(func(*ui.Spinbox) {
			windowData.Models[m.ModelID].OutputNodes = outputNodes.Value()
		})

		form3 := ui.NewForm()
		form3.SetPadded(true)
		hbox1.Append(form3, false)
		form3.Append("# Output Nodes", outputNodes, false)

		// # of epochs
		epochs := ui.NewSpinbox(0, 10000)
		epochs.SetValue(windowData.Models[m.ModelID].NumEpochs)
		epochs.OnChanged(func(*ui.Spinbox) {
			windowData.Models[m.ModelID].NumEpochs = epochs.Value()
		})

		form4 := ui.NewForm()
		form4.SetPadded(true)
		hbox1.Append(form4, false)
		form4.Append("# of Epochs", epochs, false)

		// learning rate
		lrate := ui.NewEntry()
		s := strconv.FormatFloat(windowData.Models[m.ModelID].LearningRate, 'g', -1, 64)
		lrate.SetText(s)
		lrate.OnChanged(func(*ui.Entry) {
			f, err := strconv.ParseFloat(lrate.Text(), 64)
			if err == nil {
				windowData.Models[m.ModelID].LearningRate = f
			} else {
				ui.MsgBoxError(mainwin,
					"Not a Number.",
					"Please enter a numerical value.")
				lrate.SetText("")
			}
		})

		form5 := ui.NewForm()
		form5.SetPadded(true)
		hbox1.Append(form5, false)
		form5.Append("Learning Rate", lrate, false)

		// momentum
		momentum := ui.NewEntry()
		s = strconv.FormatFloat(windowData.Models[m.ModelID].Momentum, 'g', -1, 64)
		momentum.SetText(s)
		momentum.OnChanged(func(*ui.Entry) {
			f, err := strconv.ParseFloat(momentum.Text(), 64)
			if err == nil {
				windowData.Models[m.ModelID].Momentum = f
			} else {
				ui.MsgBoxError(mainwin,
					"Not a Number.",
					"Please enter a numerical value.")
				momentum.SetText("")
			}
		})

		form6 := ui.NewForm()
		form6.SetPadded(true)
		hbox1.Append(form6, false)
		form6.Append("Momentum", momentum, false)
	} else if m.Name == "Model 2" {
		hbox1 := ui.NewHorizontalBox()
		hbox1.SetPadded(true)
		grid.Append(hbox1,
			0, 1, 1, 1,
			false, ui.AlignFill, false, ui.AlignFill)
		// layers and learning rate
		layers := ui.NewSpinbox(0, 10000)
		layers.SetValue(windowData.Models[m.ModelID].Layers)
		layers.OnChanged(func(*ui.Spinbox) {
			windowData.Models[m.ModelID].Layers = layers.Value()
		})

		form2 := ui.NewForm()
		form2.SetPadded(true)
		hbox1.Append(form2, false)
		form2.Append("numLayers", layers, false)

		lrate := ui.NewEntry()
		lrate.SetText(windowData.Models[m.ModelID].Learning)
		lrate.OnChanged(func(*ui.Entry) {
			windowData.Models[m.ModelID].Learning = lrate.Text()
		})

		form1 := ui.NewForm()
		form1.SetPadded(true)
		hbox1.Append(form1, false)
		form1.Append("learning rate", lrate, false)
	} else if m.Name == "Model 3" {
		hbox1 := ui.NewHorizontalBox()
		hbox1.SetPadded(true)
		grid.Append(hbox1,
			0, 1, 1, 1,
			false, ui.AlignFill, false, ui.AlignFill)
		// numTrees and max depth
		numTrees := ui.NewSpinbox(0, 10000)
		numTrees.SetValue(windowData.Models[m.ModelID].Trees)
		numTrees.OnChanged(func(*ui.Spinbox) {
			windowData.Models[m.ModelID].Trees = numTrees.Value()
		})

		form1 := ui.NewForm()
		form1.SetPadded(true)
		hbox1.Append(form1, false)
		form1.Append("numTrees", numTrees, false)

		maxDepth := ui.NewSpinbox(0, 10000)
		maxDepth.SetValue(windowData.Models[m.ModelID].MaxDepth)
		maxDepth.OnChanged(func(*ui.Spinbox) {
			windowData.Models[m.ModelID].MaxDepth = maxDepth.Value()
		})

		form2 := ui.NewForm()
		form2.SetPadded(true)
		hbox1.Append(form2, false)
		form2.Append("maxDepth", maxDepth, false)
	}

	form.Append("Model", alignment, false)

	return hbox
}

func generateFromState() *ui.Grid {
	grid := ui.NewGrid()
	grid.SetPadded(true)

	button := ui.NewButton("Training Data")
	entry := ui.NewEntry()
	entry.SetReadOnly(true)
	entry.SetText(windowData.TrainData)
	button.OnClicked(func(*ui.Button) {
		filename := ui.OpenFile(mainwin)
		if filename == "" {
			filename = "(cancelled)"
		}
		entry.SetText(filename)
		windowData.TrainData = filename
	})
	grid.Append(button,
		0, 0, 1, 1,
		false, ui.AlignFill, false, ui.AlignFill)
	grid.Append(entry,
		1, 0, 1, 1,
		true, ui.AlignFill, false, ui.AlignFill)

	button1 := ui.NewButton("Test Data")
	entry1 := ui.NewEntry()
	entry1.SetReadOnly(true)
	entry1.SetText(windowData.TestData)
	button1.OnClicked(func(*ui.Button) {
		filename := ui.OpenFile(mainwin)
		if filename == "" {
			filename = "(cancelled)"
		}
		entry1.SetText(filename)
		windowData.TestData = filename
	})
	grid.Append(button1,
		0, 1, 1, 1,
		false, ui.AlignFill, false, ui.AlignFill)
	grid.Append(entry1,
		1, 1, 1, 1,
		true, ui.AlignFill, false, ui.AlignFill)
	modelCount = windowData.ModelCount
	for i := 0; i < windowData.ModelCount; i++ {
		m := makeModelParam(windowData.Models[i])
		grid.Append(m,
			0, i+2, 2, 1,
			true, ui.AlignFill, false, ui.AlignFill)
	}
	return grid
}

func makeToolbar2() ui.Control {
	vbox := ui.NewVerticalBox()
	vbox.SetPadded(true)

	msggrid := ui.NewGrid()
	msggrid.SetPadded(true)
	vbox.Append(msggrid, false)

	grid := ui.NewGrid()
	grid.SetPadded(true)
	vbox.Append(grid, false)

	button := ui.NewButton("Import Config")
	button.OnClicked(func(*ui.Button) {
		filename := ui.OpenFile(mainwin)
		if filename != "" {
			file, _ := ioutil.ReadFile(filename)
			temp := UIWindow{}
			_ = json.Unmarshal([]byte(file), &temp)
			windowData = temp
			vbox.Delete(1)
			grid = generateFromState()
			vbox.Append(grid, false)
		}
	})
	msggrid.Append(button,
		0, 0, 1, 1,
		false, ui.AlignFill, false, ui.AlignFill)
	button = ui.NewButton("Add Model")
	button.OnClicked(func(*ui.Button) {
		if modelCount < 5 {
			var m ModelConfig
			m.Name = ""
			m.ModelID = modelCount
			windowData.Models[modelCount] = m
			model := makeModelParam(m)
			grid.Append(model,
				0, modelCount+2, 2, 1,
				true, ui.AlignFill, false, ui.AlignFill)
			modelCount++
			windowData.ModelCount++
		}
	})
	msggrid.Append(button,
		1, 0, 1, 1,
		false, ui.AlignFill, false, ui.AlignFill)

	button = ui.NewButton("Save Config")
	button.OnClicked(func(*ui.Button) {
		filename := ui.SaveFile(mainwin)
		file, _ := json.MarshalIndent(windowData, "", " ")
		_ = ioutil.WriteFile(filename, file, 0644)
	})
	msggrid.Append(button,
		2, 0, 1, 1,
		false, ui.AlignFill, false, ui.AlignFill)

	button = ui.NewButton("Clear All")
	button.OnClicked(func(*ui.Button) {
		temp := UIWindow{}
		windowData = temp
		windowData.Models = make([]ModelConfig, 5)
		windowData.ModelCount = 0
		vbox.Delete(1)
		grid = generateFromState()
		vbox.Append(grid, false)
	})
	msggrid.Append(button,
		3, 0, 1, 1,
		false, ui.AlignFill, false, ui.AlignFill)

	button = ui.NewButton("Run Models")
	button.OnClicked(func(*ui.Button) {
		runNN()
	})
	msggrid.Append(button,
		4, 0, 1, 1,
		false, ui.AlignFill, false, ui.AlignFill)

	button = ui.NewButton("Training Data")
	entry := ui.NewEntry()
	entry.SetReadOnly(true)
	button.OnClicked(func(*ui.Button) {
		filename := ui.OpenFile(mainwin)
		if filename == "" {
			filename = "(cancelled)"
		}
		entry.SetText(filename)
		windowData.TrainData = filename
	})
	grid.Append(button,
		0, 0, 1, 1,
		false, ui.AlignFill, false, ui.AlignFill)
	grid.Append(entry,
		1, 0, 1, 1,
		true, ui.AlignFill, false, ui.AlignFill)

	button1 := ui.NewButton("Test Data")
	entry1 := ui.NewEntry()
	entry1.SetReadOnly(true)
	button1.OnClicked(func(*ui.Button) {
		filename := ui.OpenFile(mainwin)
		if filename == "" {
			filename = "(cancelled)"
		}
		entry1.SetText(filename)
		windowData.TestData = filename
	})
	grid.Append(button1,
		0, 1, 1, 1,
		false, ui.AlignFill, false, ui.AlignFill)
	grid.Append(entry1,
		1, 1, 1, 1,
		true, ui.AlignFill, false, ui.AlignFill)

	return vbox
}

func setupUI() {
	mainwin = ui.NewWindow("libui Control Gallery", 1800, 900, true)
	windowData.Models = make([]ModelConfig, 5)
	windowData.ModelCount = 0
	mainwin.OnClosing(func(*ui.Window) bool {
		ui.Quit()
		return true
	})
	ui.OnShouldQuit(func() bool {
		mainwin.Destroy()
		return true
	})
	mainwin.SetMargined(true)
	mainwin.SetChild(makeToolbar2())

	mainwin.Show()
}

func parseCSV(path string) [][][]float64 {
	data := make([][][]float64, 0)

	// Open the file

	s := strings.Split(path, "\\")
	fixedPath := "../datasets/" + s[len(s)-1]

	csvfile, err := os.Open(fixedPath)
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
			oneEntry := [][]float64{floatarr, expected}

			data = append(data, oneEntry)
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

func runNN() {
	train := parseCSV(windowData.TrainData)
	test := parseCSV(windowData.TestData)

	for i := 0; i < modelCount; i++ {
		m := windowData.Models[i]
		//interval := (m.InputNodes - m.OutputNodes) / (m.NumHiddenLayers + 1)
		hidden := make([]int, m.NumHiddenLayers)

		for j := 0; j < m.NumHiddenLayers; j++ {
			//hidden[j] = m.InputNodes - (interval * (j + 1))
			hidden[j] = 32
		}

		if m.InputNodes == 0 {
			m.InputNodes = len(train[0][0])
		}
		fmt.Println("size of input", len(train[0][0]))
		nn0 := gonet.New(m.InputNodes, hidden, m.OutputNodes, true)
		nn1 := gonet.New(m.InputNodes, hidden, m.OutputNodes, true)
		nn2 := gonet.New(m.InputNodes, hidden, m.OutputNodes, true)
		nn3 := gonet.New(m.InputNodes, hidden, m.OutputNodes, true)
		nn4 := gonet.New(m.InputNodes, hidden, m.OutputNodes, true)
		nn5 := gonet.New(m.InputNodes, hidden, m.OutputNodes, true)
		nn6 := gonet.New(m.InputNodes, hidden, m.OutputNodes, true)

		fmt.Println("Training Default")
		timer := time.Now()
		nn0.Train(train, m.NumEpochs, m.LearningRate, m.Momentum, true)
		fmt.Printf("Runtime: %.5f seconds\n\n", time.Since(timer).Seconds())
		timer = time.Now()
		fmt.Println("Training with Decreased Epochs")
		timer = time.Now()
		nn1.Train(train, m.NumEpochs/2, m.LearningRate, m.Momentum, true)
		fmt.Printf("Runtime: %.5f seconds\n\n", time.Since(timer).Seconds())
		fmt.Println("Training with Increased Epochs")
		timer = time.Now()
		nn2.Train(train, m.NumEpochs*2, m.LearningRate, m.Momentum, true)
		fmt.Printf("Runtime: %.5f seconds\n\n", time.Since(timer).Seconds())
		fmt.Println("Training with Decreased Learning Rate")
		timer = time.Now()
		nn3.Train(train, m.NumEpochs, m.LearningRate/2, m.Momentum, true)
		fmt.Printf("Runtime: %.5f seconds\n\n", time.Since(timer).Seconds())
		fmt.Println("Training with Increased Learning Rate")
		timer = time.Now()
		nn4.Train(train, m.NumEpochs, m.LearningRate*2, m.Momentum, true)
		fmt.Printf("Runtime: %.5f seconds\n\n", time.Since(timer).Seconds())
		fmt.Println("Training with Decreased Momentum")
		timer = time.Now()
		nn5.Train(train, m.NumEpochs, m.LearningRate, m.Momentum/2, true)
		fmt.Printf("Runtime: %.5f seconds\n\n", time.Since(timer).Seconds())
		fmt.Println("Training with Increased Momentum")
		timer = time.Now()
		nn6.Train(train, m.NumEpochs, m.LearningRate, m.Momentum*2, true)
		fmt.Printf("Runtime: %.5f seconds\n\n", time.Since(timer).Seconds())

		// Predict
		totalcorrect0 := 0.0
		totalcorrect1 := 0.0
		totalcorrect2 := 0.0
		totalcorrect3 := 0.0
		totalcorrect4 := 0.0
		totalcorrect5 := 0.0
		totalcorrect6 := 0.0
		for i := 0; i < len(test); i++ {
			// fmt.Println("expected", MinMax(test[i][1]))
			// fmt.Println("predicted", MinMax(nn.Predict(test[i][0])))
			// s := fmt.Sprintf("%d, %d | ", MinMax(test[i][1]), MinMax(nn0.Predict(test[i][0])))
			// fmt.Print(s)
			// if i%15 == 0 {
			//	 fmt.Println()
			// }
			if MinMax(test[i][1]) == MinMax(nn0.Predict(test[i][0])) {
				totalcorrect0 += 1.0
			}
			if MinMax(test[i][1]) == MinMax(nn1.Predict(test[i][0])) {
				totalcorrect1 += 1.0
			}
			if MinMax(test[i][1]) == MinMax(nn2.Predict(test[i][0])) {
				totalcorrect2 += 1.0
			}
			if MinMax(test[i][1]) == MinMax(nn3.Predict(test[i][0])) {
				totalcorrect3 += 1.0
			}
			if MinMax(test[i][1]) == MinMax(nn4.Predict(test[i][0])) {
				totalcorrect4 += 1.0
			}
			if MinMax(test[i][1]) == MinMax(nn5.Predict(test[i][0])) {
				totalcorrect5 += 1.0
			}
			if MinMax(test[i][1]) == MinMax(nn6.Predict(test[i][0])) {
				totalcorrect6 += 1.0
			}
		}
		output0 := fmt.Sprintf("Default Percent correct: %.2f %s\n", totalcorrect0/float64(len(test))*100.0, "%")
		output1 := fmt.Sprintf("Decrease Epochs Percent correct: %.2f %s\n", totalcorrect1/float64(len(test))*100.0, "%")
		output2 := fmt.Sprintf("Increase Epochs Percent correct: %.2f %s\n", totalcorrect2/float64(len(test))*100.0, "%")
		output3 := fmt.Sprintf("Decrease Learning Rate Percent correct: %.2f %s\n", totalcorrect3/float64(len(test))*100.0, "%")
		output4 := fmt.Sprintf("Increase Learning Rate Percent correct: %.2f %s\n", totalcorrect4/float64(len(test))*100.0, "%")
		output5 := fmt.Sprintf("Decrease Momentum Percent correct: %.2f %s\n", totalcorrect5/float64(len(test))*100.0, "%")
		output6 := fmt.Sprintf("Increase Momentum Percent correct: %.2f %s\n", totalcorrect6/float64(len(test))*100.0, "%")
		fmt.Print(output0, output1, output2, output3, output4, output5, output6)
	}
	// // Save the model
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

func main() {
	ui.Main(setupUI)
}
