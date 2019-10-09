/******************************** 
*
*********************************/
async function getData() {
    const url  = 'https://raw.githubusercontent.com/kuc-arc-f/tfjs_start2/master/dat/outout.json';
    const carsDataReq = await fetch( url );  
    const carsData = await carsDataReq.json();  
//console.log(carsData)
    const cleaned = carsData.map(car => ({
        hnum: car.hnum,
        no: car.no,
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
    x: d.no,
    y: d.hnum,
}));
//console.log( "len="+ values.length )
//  console.log( values )

tfvis.render.scatterplot(
    {name: 'Horsepower v MPG'},
    {values}, 
    {
    xLabel: 'xs',
    yLabel: 'hnum',
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
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
//    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
//    model.add(tf.layers.dense({units: 1, useBias: true}));
    return model;
}
/******************************** 
*
*********************************/
function convertToTensor(data) {
return tf.tidy(() => {
//	tf.util.shuffle(data);	
    const inputs = data.map(d => d.no)   //X- axis
    const labels = data.map(d => d.hnum ); //Y-axis
//	console.log(labels)
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
async function trainModel(model, inputs, labels, tensorData, input_size) {
    const {inputMax, inputMin, labelMin, labelMax} = tensorData
//console.log( input_size );
    // Prepare the model for training.  
    // model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
    const column_len = input_size
    model.compile({
//        optimizer: tf.train.adam(),
        optimizer: 'sgd',
        loss: 'meanSquaredError',
//        metrics: ['mse'],
    });
    const batchSize = column_len -5;
    const epochs = 200;
//    const epochs = 200;
    //
    //await model.fit(xs, ys, {epochs: 1000});
    return await model.fit(inputs, labels, {
		batchSize,
		epochs,
		shuffle: true,
		callbacks: tfvis.show.fitCallbacks(
		{ name: 'Training Performance' },
		['loss', 'mse'], 
		{ height: 200, callbacks: ['onEpochEnd'] }
		)
	});


}