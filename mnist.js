class Mnist {

	constructor(imgfile){
		this.filepath = imgfile;
		
		console.log("Created a stock object with filepath "+ this.filepath);
		this.tfTrainingNew();

	}

	tfTrainingOld() {
		let self = this;
		this.readFile().then((result) =>{

			let n_stocks = self.rawData.training.length -1;
			let n_neurons_1 = 1024;
			let n_neurons_2 = 512;
			let n_neurons_3 = 256;
			let n_neurons_4 = 128;
			let n_target = self.rawData.training.length -1;
			let X = tf.tensor(self.rawData.training.slice(0,-2));
			let Y = tf.tensor(self.rawData.training.slice(1, -1));
			let W_hidden_1 = tf.variable(tf.truncatedNormal([n_stocks, n_neurons_1]));
			let bias_hidden_1 = tf.variable(tf.zeros([n_neurons_1]));
			let W_hidden_2 = tf.variable(tf.truncatedNormal([n_neurons_1, n_neurons_2]));
			let bias_hidden_2 = tf.variable(tf.zeros([n_neurons_2]));
			let W_hidden_3 = tf.variable(tf.truncatedNormal([n_neurons_2, n_neurons_3]));
			let bias_hidden_3 = tf.variable(tf.zeros([n_neurons_3]));
			let W_hidden_4 = tf.variable(tf.truncatedNormal([n_neurons_3, n_neurons_4]));
			let bias_hidden_4 = tf.variable(tf.zeros([n_neurons_4]));
			let W_out = tf.variable(tf.truncatedNormal([n_neurons_4, n_target]));
			let bias_out = tf.variable(tf.zeros([n_target]));

			const learningRate = 0.5;
			const optimizer = tf.train.adamax(learningRate);

			tf.tidy(function(){

				let out = function(X) {
					let hidden_1 = tf.relu(tf.add(tf.matMul(X, W_hidden_1, true), bias_hidden_1));
					let hidden_2 = tf.relu(tf.add(tf.matMul(hidden_1, W_hidden_2), bias_hidden_2));
					let hidden_3 = tf.relu(tf.add(tf.matMul(hidden_2, W_hidden_3), bias_hidden_3));
					let hidden_4 = tf.relu(tf.add(tf.matMul(hidden_3, W_hidden_4), bias_hidden_4));
				
					return tf.transpose(tf.add(tf.matMul(hidden_4, W_out), bias_out));
				}

				console.log("Input:", X, " Output:", Y );
				optimizer.minimize(()=> tf.losses.meanSquaredError(out(X), Y));
			});
		});
	}

	tfTrainingNew() {
		let self = this;
		this.readFile().then(async function(result){

			let n_neurons_1 = 1024;
			let n_neurons_2 = 512;
			let n_neurons_3 = 256;
			let n_neurons_4 = 128;  

			let X = tf.tensor2d(self.rawData.training.slice(0,-1));
			let Y = tf.tensor1d(self.rawData.training.slice(1).map((ele) => ele[1]));

			const model = tf.sequential();

			model.add(tf.layers.dense({
				inputShape: [2],
				activation: "relu",
				units: n_neurons_1
			}));

			model.add(tf.layers.dense({
				inputShape: [n_neurons_1],
				activation: "relu",
				units: n_neurons_2
			}));

			model.add(tf.layers.dense({
				inputShape: [n_neurons_2],
				activation: "relu",
				units: n_neurons_3
			}));

			model.add(tf.layers.dense({
				inputShape: [n_neurons_3],
				activation: "relu",
				units: n_neurons_4
			}));

			model.add(tf.layers.dense({
				inputShape: [n_neurons_4],
				activation: "relu",
				units: 1
			}));

			model.compile({
				loss: "meanSquaredError",
				optimizer: tf.train.adamax(0.3)
			});

			const startTime = Date.now();
			model.fit(X, Y, {epochs: 10}).then((history) => {
					console.log("DONE!", Date.now() - startTime);
					console.log(history);
					let testingX = tf.tensor2d(self.rawData.testing);
					model.predict(testingX).print();

					let download = confirm("Download?");
					if(download){
						model.save('downloads://my-model-1');
					}
				});
		});
	}


	tfTrainingLoad() {
		let self = this;

		return this.readFile().then(async (result) => {
			let testingX = tf.tensor2d(self.rawData.testing.map((ele) => ele[0]));

			const model = await tf.loadModel('https://photo.recognize.tech/tensorflow/my-model-1.json');
			model.predict(testingX).print();

		});
	}

	get data(){
		return this.readFile().then((result) =>{
			return result;
		});
	}

	readFile() {
		let self = this;

		return new Promise((resolve, reject) => {

			let xhttp = new XMLHttpRequest();
			let filepath = this.filepath;

			xhttp.onreadystatechange = function() {
				if(this.readyState === 4 && this.status === 200){
					let data = this.responseText;
					let dataSplitByLine = data.split("\n");
					let tmpRawData = {training: [], testing: []};
					let trainMax = Number.MIN_VALUE;
					let trainMin = Number.MAX_VALUE;
					
					for(let i=1; i < dataSplitByLine.length; i++){
						let dataValues = dataSplitByLine[i].split(",");
						let tmpRawSubData = [];

						if(dataValues.every((value) => value !== "null") && !isNaN(dataValues[1])){
							tmpRawSubData.push(i);
							tmpRawSubData.push(parseFloat(dataValues[1]));

							// Saving data to training set and testing set respectively
							if(i < 3458){
								tmpRawData.training.push(tmpRawSubData);
							} else {
								tmpRawData.testing.push(tmpRawSubData);
							}

							if(tmpRawSubData[1] > trainMax){
								trainMax = tmpRawSubData[1];
							}

							if(tmpRawSubData[1] < trainMin){
								trainMin = tmpRawSubData[1];
							}
						}
					}

					// Normalize the data
					for(let i=0; i < tmpRawData.training.length; i++){
						tmpRawData.training[i][1] = (tmpRawData.training[i][1] - trainMin) / (trainMax - trainMin);
					}

					for(let i=0; i < tmpRawData.testing.length; i++){
						tmpRawData.testing[i][1] = (tmpRawData.testing[i][1] - trainMin) / (trainMax - trainMin);
					}

					self.rawData = tmpRawData;

					resolve(tmpRawData);
				}
			}
			xhttp.open("GET", this.filepath, true);
			xhttp.send();

		});
	}
}


let stock = new Stock("0700.HK.new.csv");
let trainSourceElement = document.querySelector("#trainsource");
let testSourceElement = document.querySelector("#testsource");

stock.data.then((result) => {

	// ==================== HTML Display =====================
	let trace = { x: [], y: [], type: 'Scatter + Lines'};
	
	let htmlData = "";
	htmlData += "Training Data <br>";

	for(let ele of result.training){
		trace.x.push(ele[0]);
		trace.y.push(ele[1]);
		htmlData += String(ele[0]);
		htmlData += " : ";
		htmlData += String(ele[1]);
		htmlData += "<br>"; 
	}

	trainSourceElement.innerHTML = htmlData;
	htmlData = "";

	htmlData += "Testing Data <br>";

	for(let ele of result.testing){
		trace.x.push(ele[0]);
		trace.y.push(ele[1]);
		htmlData += String(ele[0]);
		htmlData += " : ";
		htmlData += String(ele[1]);
		htmlData += "<br>"; 
	}

	testSourceElement.innerHTML = htmlData;

	let layout = {xaxis:{zeroline: false}};

	Plotly.newPlot("sourceGraph", [trace], layout);

});