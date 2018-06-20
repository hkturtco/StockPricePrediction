let xs = [];
let ys = [];

let m, b;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);
 

function setup(){
	createCanvas(300, 500);
	
	m = tf.variable(tf.scalar(random(1)));
	b = tf.variable(tf.scalar(random(1)));
}	

function loss(pred, labels){
	return pred.sub(labels).square().mean();
}

function predict(tfxs){
	const tsys = tfxs.mul(m).add(b);

	return tsys;
}

function mousePressed(){

	let x = map(mouseX, 0, width, 0, 1);
	let y = map(mouseY, 0, height, 1, 0);
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
		let px = map(xs[i], 0, 1, 0, width);
		let py = map(ys[i], 0, 1, height, 0);
		point(px, py);
	}

	tf.tidy(function(){
		let xsG = [0,1];
		const tfxsG = tf.tensor1d(xsG);
		const tfysG = predict(tfxsG);
		let ysR = tfysG.dataSync();

		let x1 = map(xsG[0], 0, 1, 0, width);
		let x2 = map(xsG[1], 0, 1, 0, width);
		let y1 = map(ysR[0], 0, 1, height,0);
		let y2 = map(ysR[1], 0, 1, height,0);

		line(x1, y1, x2, y2);
	});


	console.log(tf.memory().numTensors);

}