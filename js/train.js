/******************************** 
*
*********************************/
 async function getData() {
	const url  = 'https://raw.githubusercontent.com/kuc-arc-f/tfjs_start2/master/dat/outout.json';
	const carsDataReq = await fetch( url );  
	const carsData = await carsDataReq.json();  
//console.log(carsData)
	const cleaned = carsData.map(car => ({
		no: car.no,
		hnum: car.hnum,
	}))

	return cleaned;
}
/******************************** 
*
*********************************/
async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));
  //console.log( "len="+ values.length )
//  console.log( values )

  tfvis.render.scatterplot(
    {name: 'Horsepower v MPG'},
    {values}, 
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
  // More code will be added below
}
/******************************** 
*
*********************************/
function createModel() {
  // Create a sequential model
  const model = tf.sequential(); 
  // Add a single hidden layer
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
  // Add an output layer
  model.add(tf.layers.dense({units: 1, useBias: true}));
  return model;
}
/******************************** 
*
*********************************/
function convertToTensor(data) {
  return tf.tidy(() => {
//	tf.util.shuffle(data);
	
    const inputs = data.map(d => d.hnum)
    const labels = data.map(d => d.no );

//	console.log(labels)
//	return
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //  Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();  
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });  
}
/******************************** 
*
*********************************/
async function trainModel(model, inputs, labels, tensorData) {
	const {inputMax, inputMin, labelMin, labelMax} = tensorData
//console.log( inputMax);
	// Prepare the model for training.  
	model.compile({
		optimizer: tf.train.adam(),
		loss: tf.losses.meanSquaredError,
		metrics: ['mse'],
	});

	const batchSize = 32;
	const epochs = 50;
//	const epochs = 30;

//	return model.fit(inputs,labels,{epochs: epochs });
	 model.fit(inputs,labels,{epochs: epochs });
	console.log('#fit-complete');
	//pred
	const [xs, preds] = tf.tidy(() => {
		const xs = tf.linspace(0, 1, 100);      
		const preds = model.predict(xs.reshape([100, 1]));      

		/* 
		const unNormXs = xs
			.mul(inputMax.sub(inputMin))
			.add(inputMin);
		*/
		const unNormXs = xs
			.mul(labelMax.sub(labelMin))
			.add(labelMin);
/* 
		const unNormPreds = preds
			.mul(labelMax.sub(labelMin))
			.add(labelMin);
*/
		const unNormPreds = preds
		.mul(inputMax.sub(inputMin))
		.add(inputMin);

		// Un-normalize the data
		return [unNormXs.dataSync(), unNormPreds.dataSync()];
	});
	console.log(xs );
	console.log(preds );

	return

}