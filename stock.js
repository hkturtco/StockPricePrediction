class Stock {

	constructor(stockfile){
		this.filepath = stockfile;
		console.log("Created a stock object with filepath "+ this.filepath);
	}

	get data(){
		return this.readFile().then((result) =>{
			//============ Processing data =================
			let dataSplitByLine = result.split("\n");
			let htmlDisplay = "";
			
			for(let i=0; i < dataSplitByLine.length; i++){
				dataSplitByLine[i] = dataSplitByLine[i].split(",");
				htmlDisplay += (dataSplitByLine[i].join(" ") + "<br>");
			}

			this.rawData = dataSplitByLine;
			return htmlDisplay;
		});
	}

	readFile(){
		return new Promise((resolve, reject) => {

			let xhttp = new XMLHttpRequest();
			let filepath = this.filepath;

			xhttp.onreadystatechange = function() {
				if(this.readyState === 4 && this.status === 200){
					let data = this.responseText;
					resolve(data);
				}
			}
			xhttp.open("GET", this.filepath, true);
			xhttp.send();

		});

	}
}


let stock = new Stock("0700.HK.csv");
let sourceElement = document.querySelector("#source");

stock.data.then((result) => {
	// ==================== HTML Display =====================

	sourceElement.innerHTML = result;

});