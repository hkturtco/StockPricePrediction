class Stock {

	constructor(stockfile){
		this.filepath = stockfile;
		
		console.log("Created a stock object with filepath "+ this.filepath);
	}

	tfTrainingNew() {
		let self = this;

		return new Promise((resolve, reject) => {
			self.readFile().then(async function(result){


				let xInput = self.rawData.training.slice(0, -1).map((ele) => [[ele[0][1]]]);
				let yInput = self.rawData.training.slice(1).map((ele) => ele[0][1]);
				let X = tf.tensor3d(xInput);
				let Y = tf.tensor1d(yInput);

				const model = tf.sequential();

				const lstm = tf.layers.lstm({units: 10});
				const input = tf.input({shape: [1, 1]});
				const output = lstm.apply(input);

				const dense = tf.layers.dense({units: 1});
				const activation = tf.layers.activation({activation: 'linear'});


				model.add(lstm);
				model.add(dense);
				model.add(activation);

				model.compile({
					loss: "meanSquaredError",
					optimizer: 'adam'
				});

				const startTime = Date.now();

				model.fit(X, Y, {epochs: 5, batchSize: 1}).then((history) => {
						
					console.log("DONE!", Date.now() - startTime);
					console.log(history);

					let testingX = tf.tensor3d(self.rawData.testing.map((ele) => [[ele[0][1]]]));
					resolve(model.predict(testingX).dataSync());

					let download = confirm("Download?");
					if(download){
						model.save('downloads://my-model-1');
					}
				});

			});
		});
	}

	tfModelLoad() {
		let self = this;

		return this.readFile().then(async (result) => {
			let testingX = tf.tensor2d(self.rawData.testing.map((ele) => ele[0]));

			const model = await tf.loadModel('https://photo.recognize.tech/tensorflow/my-model-1.json');
			model.predict(testingX).print();

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
							if(i < 3300){
								tmpRawData.training.push([tmpRawSubData]);
							} else {
								tmpRawData.testing.push([tmpRawSubData]);
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
						tmpRawData.training[i][0][1] = (tmpRawData.training[i][0][1] - trainMin) / (trainMax - trainMin);
					}

					for(let i=0; i < tmpRawData.testing.length; i++){
						tmpRawData.testing[i][0][1] = (tmpRawData.testing[i][0][1] - trainMin) / (trainMax - trainMin);
					}

					self.rawData = tmpRawData;

					resolve(tmpRawData);
				}
			}
			xhttp.open("GET", this.filepath, true);
			xhttp.send();

		});
	}

	get data(){
		let readFile = this.readFile();
		let trained = this.tfTrainingNew();

		return Promise.all([readFile, trained]);
	}
}


let stock = new Stock("0700.HK.new.csv");
let trainSourceElement = document.querySelector("#trainsource");
let testSourceElement = document.querySelector("#testsource");

stock.data.then((result) => {

	// ==================== HTML Display =====================
	let trace = { x: [], y: [], type: 'Scatter + Lines'};
	let resultTrace = { x: [], y: [], type: 'Scatter + Lines'};

	let htmlData = "";
	htmlData += "Training Data <br>";

	for(let ele of result[0].training){
		trace.x.push(ele[0][0]);
		trace.y.push(ele[0][1]);
		htmlData += String(ele[0][0]);
		htmlData += " : ";
		htmlData += String(ele[0][1]);
		htmlData += "<br>"; 
	}

	trainSourceElement.innerHTML = htmlData;
	htmlData = "";

	htmlData += "Testing Data <br>";

	let dayForResult = null;
	let i = 0;

	for(let ele of result[0].testing){

		if(dayForResult === null){
			firstDayForResult = Number(ele[0][0]);
			dayForResult = firstDayForResult;
		} else {
			i = i + 1;
			dayForResult = firstDayForResult + i;
		}
		trace.x.push(ele[0][0]);
		trace.y.push(ele[0][1]);

		resultTrace.x.push(dayForResult+1);
		resultTrace.y.push(result[1][i]);

		htmlData += String(ele[0][0]);
		htmlData += " : ";
		htmlData += String(ele[0][1]);
		htmlData += " => ";
		htmlData += String(dayForResult+1);
		htmlData += " : ";
		htmlData += String(result[1][i]);
		htmlData += "<br>"; 
	}

	testSourceElement.innerHTML = htmlData;
	console.log(resultTrace);

	let layout = {xaxis:{zeroline: false}};

	Plotly.newPlot("sourceGraph", [trace, resultTrace], layout);

});