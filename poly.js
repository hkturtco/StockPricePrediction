let xs = [];
let ys = [];

let a, b, c;

const learningRate = 0.5;
const optimizer = tf.train.adam(learningRate);
 

function setup(){
	createCanvas(300, 500);
	
	a = tf.variable(tf.scalar(random(-1,1)));
	b = tf.variable(tf.scalar(random(-1,1)));
	c = tf.variable(tf.scalar(random(-1,1)));
}	

function loss(pred, labels){
	return pred.sub(labels).square().mean();
}

function predict(tfxs){
	const tsys = tfxs.pow(tf.scalar(2)).mul(a).add(tfxs.mul(b)).add(c);

	return tsys;
}

function mousePressed(){

	let x = map(mouseX, 0, width, -1, 1);
	let y = map(mouseY, 0, height, 1, -1);
	xs.push(x);
	ys.push(y);
}

// loop function in p5
function draw(){

	tf.tidy(function(){
		const tfys = tf.tensor1d(ys);
		const tfxs = tf.tensor1d(xs);

		optimizer.minimize(()=> loss(predict(tfxs), tfys));
	});

	background(0);

	stroke(200);
	strokeWeight(7);
	for (let i = 0; i < xs.length; i++) {
		let px = map(xs[i], -1, 1, 0, width);
		let py = map(ys[i], -1, 1, height, 0);
		point(px, py);
	}

	tf.tidy(function(){
		const xsG = [];

		for(let x= -1; x <= 1 ; x+= 0.05){
			xsG.push(x);
		}

		const tfxsG = tf.tensor1d(xsG);
		const tfysG = predict(tfxsG);
		let ysR = tfysG.dataSync();


		beginShape();
		noFill();
		stroke(255);
		strokeWeight(2);
		for(let i = 0; i < xsG.length; i++){
			let x = map(xsG[i], -1, 1, 0, width);
			let y = map(ysR[i], -1, 1, height,0);
			vertex(x, y);
		}

		endShape();
	});


	//console.log(tf.memory().numTensors);

}