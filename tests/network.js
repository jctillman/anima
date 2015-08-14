var Network = require('../network/network');
var expect = require('chai').expect;
var mnistReader = require('./mnist_reader');
var _ = require('lodash');

describe('Testing network, that it creates networks and feeds forward basically right', function(){

	xdescribe('It can make a fully-connected network, with as many layers as specified, of the dimensions specified', function(){

		var neuronOptions = {
			typeOfNeuron: 'tanh',
			randomness: 'flatProportionateZero',
			cost: 'squaredError'
		};

		it('Can make a single-dimensional network', function(){
			var NN = new Network([
				{neuronOptions: neuronOptions, pattern: {dimensions: [32]}},
				{neuronOptions: neuronOptions, pattern: {type: 'full', dimensions: [7]}},
				{neuronOptions: neuronOptions, pattern: {type: 'full', dimensions: [3]}},
			]);
			//console.log(NN)
			expect(NN.layers[0].dimensions).to.eql([32]);
			expect(NN.layers[1].dimensions).to.eql([7]);
			expect(NN.layers[2].dimensions).to.eql([3]);
			expect(NN.layers[0].neurons.length).to.eql(32);
			expect(NN.layers[1].neurons.length).to.eql(7);
			expect(NN.layers[1].neurons[0].neuron.connections.length).to.eql(32);
			expect(NN.layers[1].neurons[1].neuron.connections.length).to.eql(32);
			expect(NN.layers[2].neurons.length).to.eql(3);
			expect(NN.layers[2].neurons[0].neuron.connections.length).to.eql(7);
			expect(NN.layers[2].neurons[1].neuron.connections.length).to.eql(7);
		});

		it('Can make a multi-dimensional network', function(){
			var NN = new Network([
				{neuronOptions: neuronOptions, pattern: {dimensions: [10,10]}},
				{neuronOptions: neuronOptions, pattern: {type: 'full', dimensions: [5,5]}},
				{neuronOptions: neuronOptions, pattern: {type: 'full', dimensions: [2]}},
			]);
			expect(NN.layers[0].dimensions).to.eql([10,10]);
			expect(NN.layers[1].dimensions).to.eql([5,5]);
			expect(NN.layers[2].dimensions).to.eql([2]);
			expect(NN.layers[0].neurons.length).to.eql(100);
			expect(NN.layers[1].neurons.length).to.eql(25);
			expect(NN.layers[1].neurons[0].neuron.connections.length).to.eql(100);
			expect(NN.layers[1].neurons[1].neuron.connections.length).to.eql(100);
			expect(NN.layers[2].neurons.length).to.eql(2);
			expect(NN.layers[2].neurons[0].neuron.connections.length).to.eql(25);
			expect(NN.layers[2].neurons[1].neuron.connections.length).to.eql(25);
		});

	});

	xdescribe('It can make a partially connected network without shared weights, with as many layers as specified, of the dimensions specified', function(){

		var neuronOptions = {
			typeOfNeuron: 'tanh',
			randomness: 'flatProportionateZero',
			cost: 'squaredError'
		};

		it('Can make a single-dimensional, partially connected network', function(){

			var NN = new Network([
				{neuronOptions: neuronOptions, pattern: {dimensions: [11]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', field: [3], stride:[2]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', field: [3], stride:[1]}}
			]);

			expect(NN.layers[0].dimensions).to.eql([11]);
			expect(NN.layers[1].dimensions).to.eql([5]);
			expect(NN.layers[2].dimensions).to.eql([3]);
			expect(NN.layers[0].neurons.length).to.eql(11);
			expect(NN.layers[1].neurons.length).to.eql(5);
			expect(NN.layers[1].neurons[0].neuron.connections.length).to.eql(3);
			expect(NN.layers[1].neurons[1].neuron.connections.length).to.eql(3);
			expect(NN.layers[2].neurons.length).to.eql(3);
			expect(NN.layers[2].neurons[0].neuron.connections.length).to.eql(3);
			expect(NN.layers[2].neurons[1].neuron.connections.length).to.eql(3);
		});

		it('Can make a multi-dimensional, partially-connected network', function(){

			var NN = new Network([
				{neuronOptions: neuronOptions, pattern: {dimensions: [11,11]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', field: [3,3], stride:[2,2]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', field: [3,3], stride:[1,1]}}
			]);

			expect(NN.layers[0].dimensions).to.eql([11,11]);
			expect(NN.layers[1].dimensions).to.eql([5,5]);
			expect(NN.layers[2].dimensions).to.eql([3,3]);
			expect(NN.layers[0].neurons.length).to.eql(121);
			expect(NN.layers[1].neurons.length).to.eql(25);
			expect(NN.layers[1].neurons[0].neuron.connections.length).to.eql(9);
			expect(NN.layers[1].neurons[1].neuron.connections.length).to.eql(9);
			expect(NN.layers[2].neurons.length).to.eql(9);
			expect(NN.layers[2].neurons[0].neuron.connections.length).to.eql(9);
			expect(NN.layers[2].neurons[1].neuron.connections.length).to.eql(9);

		});

		//Whammo, passed on the first try.  Uh-huh, uh-huh.
		it('Can make a multi-dimensional, partially-connected network with alternate resolutions in different directions', function(){
			var NN = new Network([
				{neuronOptions: neuronOptions, pattern: {dimensions: [11,11]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', field: [3,3], stride:[2,1]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', field: [3,3], stride:[1,1]}}
			]);
			expect(NN.layers[0].dimensions).to.eql([11,11]);
			expect(NN.layers[1].dimensions).to.eql([5,9]);
			expect(NN.layers[2].dimensions).to.eql([3,7]);
			expect(NN.layers[0].neurons.length).to.eql(121);
			expect(NN.layers[1].neurons.length).to.eql(45);
			expect(NN.layers[1].neurons[0].neuron.connections.length).to.eql(9);
			expect(NN.layers[1].neurons[1].neuron.connections.length).to.eql(9);
			expect(NN.layers[2].neurons.length).to.eql(21);
			expect(NN.layers[2].neurons[0].neuron.connections.length).to.eql(9);
			expect(NN.layers[2].neurons[1].neuron.connections.length).to.eql(9);
		});

		it('Can make a multi-dimensional, partially-connected network with extra freakin layers and shared weights', function(){
			var NN = new Network([
				{neuronOptions: neuronOptions, pattern: {dimensions: [11,11]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', field: [3,3], stride:[2,2], depth: [5]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', field: [3,3,5], stride:[1,1,1],}}
			]);
			expect(NN.layers[0].dimensions).to.eql([11,11]);
			expect(NN.layers[1].dimensions).to.eql([5,5,5]);
			expect(NN.layers[2].dimensions).to.eql([3,3,1]);
			expect(NN.layers[0].neurons.length).to.eql(121);
			expect(NN.layers[1].neurons.length).to.eql(125);
			expect(NN.layers[1].neurons[0].neuron.connections.length).to.eql(9);
			expect(NN.layers[1].neurons[1].neuron.connections.length).to.eql(9);
			expect(NN.layers[2].neurons.length).to.eql(9);
			expect(NN.layers[2].neurons[0].neuron.connections.length).to.eql(9*5);
			expect(NN.layers[2].neurons[1].neuron.connections.length).to.eql(9*5);

		});

	});

	xdescribe('Testing initialization, activation, propogation, adjusting, in general terms', function(){

		var neuronOptions = {
			typeOfNeuron: 'tanh',
			randomness: 'flatProportionateZero',
			cost: 'squaredError'
		};

		it('Can do all of that with a multi-dimensional network', function(){
			var NN = new Network([
				{neuronOptions: neuronOptions, pattern: {dimensions: [10,10]}},
				{neuronOptions: neuronOptions, pattern: {type: 'full', dimensions: [5,5]}},
				{neuronOptions: neuronOptions, pattern: {type: 'full', dimensions: [2]}},
			]);

			//Yeah
			NN.init();

			//Can we propogate through?
			var activator = [];
			for(var x = 0; x < 100; x++){ activator.push(1);}
			var result = NN.activation(activator)
		    for(var x = 0; x < result.length; x++){
		    	expect(!isNaN(result[x])).to.equal(true)	
		    }

			NN.propogate([0,0,0]);
		    NN.adjust(0.01);
		    NN.getDeltas();
		    NN.applyDeltas(0.01);

		    //Can we propogate after doing all that ?
			var activator = [];
			for(var x = 0; x < 100; x++){ activator.push(1);}
			var result = NN.activation(activator)
		    for(var x = 0; x < result.length; x++){
		    	expect(!isNaN(result[x])).to.equal(true)	
		    }
			



		});

		it('Can do all of that with a multi-dimensional, partially-connected network', function(){

			var NN = new Network([
				{neuronOptions: neuronOptions, pattern: {dimensions: [11,11]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', field: [3,3], stride:[2,2]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', field: [3,3], stride:[1,1]}}
			]);

			//Initialize
			NN.init();

			//Can we propogate through?
			var activator = [];
			for(var x = 0; x < 121; x++){ activator.push(1);}
			var result = NN.activation(activator)
		    for(var x = 0; x < result.length; x++){
		    	expect(!isNaN(result[x])).to.equal(true)	
		    }

			NN.propogate([0,0,0]);
		    NN.adjust(0.01);
		    NN.getDeltas();
		    NN.applyDeltas(0.01)

		    //Can we propogate after doing all that ?
			var activator = [];
			for(var x = 0; x < 121; x++){ activator.push(1);}
			var result = NN.activation(activator)
		    for(var x = 0; x < result.length; x++){
		    	expect(!isNaN(result[x])).to.equal(true)	
		    }
			


		});

		it('Can do all that with a make a single-dimensional, partially-connected network with extra freakin layers and shared weights', function(){
			var NN = new Network([
				{neuronOptions: neuronOptions, pattern: {dimensions: [11]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', field: [3], stride:[2], depth: [5]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', field: [3,5], stride:[1,1], depth:[1]}}
			]);

			//Init
			NN.init();

			expect(NN.layers[0].dimensions).to.eql([11]);
			expect(NN.layers[1].dimensions).to.eql([5,5]);
			expect(NN.layers[2].dimensions).to.eql([3,1,1]);
			expect(NN.layers[0].neurons.length).to.eql(11);
			expect(NN.layers[1].neurons.length).to.eql(25);
			expect(NN.layers[1].neurons[0].neuron.connections.length).to.eql(3);
			expect(NN.layers[1].neurons[1].neuron.connections.length).to.eql(3);
			expect(NN.layers[2].neurons.length).to.eql(3);
			expect(NN.layers[2].neurons[0].neuron.connections.length).to.eql(3*5);
			expect(NN.layers[2].neurons[1].neuron.connections.length).to.eql(3*5);

			//Can we propogate through?
			var activator = [];
			for(var x = 0; x < 11; x++){ activator.push(1);}
			var result = NN.activation(activator);

			//Everything should be equal, because of the shared weights, when the activation is the same.
			var m = result[0];
			for (var x = 0; x < result.length; x++){
				expect(!isNaN(result[x])).to.equal(true)
				expect(result[x] == m).to.equal(true)
			}

			//Can we propogate through different values?
			var activator = [];
			for(var x = 0; x < 11; x++){ activator.push(Math.random());}
			var result = NN.activation(activator);

			//Everything should be different equal, because of the shared weights, when the activation is different.
			var m = result[0];
			for (var x = 1; x < result.length; x++){
				expect(!isNaN(result[x])).to.equal(true)
				expect(result[x] != m).to.equal(true)
			}
			
			//Are the neurons really sharing weights?
			expect(NN.layers[1].neurons[0].neuron.sv === NN.layers[1].neurons[5].neuron.sv).to.equal(true)
			expect(NN.layers[1].neurons[5].neuron.sv === NN.layers[1].neurons[10].neuron.sv).to.equal(true)
			expect(NN.layers[1].neurons[10].neuron.sv === NN.layers[1].neurons[15].neuron.sv).to.equal(true)

			NN.propogate([0,0,0]);
		    NN.adjust(0.01);
		    NN.getDeltas();
		    NN.applyDeltas(0.01)

		    //Can we propogate after doing all that ?
			var activator = [];
			for(var x = 0; x < 100; x++){ activator.push(1);}
			var result = NN.activation(activator)
		    for(var x = 0; x < result.length; x++){
		    	expect(!isNaN(result[x])).to.equal(true)	
		    }
						
		});

		it('Can do all that with a multi-dimensional, partially-connected network with extra freakin layers and shared weights', function(){
			var NN = new Network([
				{neuronOptions: neuronOptions, pattern: {dimensions: [11,11]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', field: [3,3], stride:[2,2], depth: [5]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', field: [3,3,5], stride:[1,1,1], depth: [1],}}
			]);
			expect(NN.layers[0].dimensions).to.eql([11,11]);
			expect(NN.layers[1].dimensions).to.eql([5,5,5]);
			expect(NN.layers[2].dimensions).to.eql([3,3,1,1]);
			expect(NN.layers[0].neurons.length).to.eql(121);
			expect(NN.layers[1].neurons.length).to.eql(125);
			expect(NN.layers[1].neurons[0].neuron.connections.length).to.eql(9);
			expect(NN.layers[1].neurons[1].neuron.connections.length).to.eql(9);
			expect(NN.layers[2].neurons.length).to.eql(9);
			expect(NN.layers[2].neurons[0].neuron.connections.length).to.eql(9*5);
			expect(NN.layers[2].neurons[1].neuron.connections.length).to.eql(9*5);

			NN.init();
			//Can we propogate through?
			var activator = [];
			for(var x = 0; x < 121; x++){ activator.push(1);}
			var result = NN.activation(activator);

			//Everything should be equal, because of the shared weights, when the activation is the same.
			var m = result[0];
			for (var x = 0; x < result.length; x++){
				expect(!isNaN(result[x])).to.equal(true)
				expect(result[x] == m).to.equal(true)
			}

			//Can we propogate through different values?
			var activator = [];
			for(var x = 0; x < 121; x++){ activator.push(Math.random());}
			var result = NN.activation(activator);

			//Everything should be different equal, because of the shared weights, when the activation is different.
			var m = result[0];
			for (var x = 1; x < result.length; x++){
				expect(!isNaN(result[x])).to.equal(true)
				expect(result[x] != m).to.equal(true)
			}

			NN.propogate([0,0,0]);
		    NN.adjust(0.01);
		    NN.getDeltas();
		    NN.applyDeltas(0.01);

		    //Can we propogate after doing all that ?
			var activator = [];
			for(var x = 0; x < 121; x++){ activator.push(1);}
			var result = NN.activation(activator)
		    for(var x = 0; x < result.length; x++){
		    	expect(!isNaN(result[x])).to.equal(true)	
		    }

		});


	});

	describe('Can it handle some *real* data correctly?', function(){

		this.timeout(50000)

		it('Can use a basic fully-connected network to distinguish 1s and 0s MNIST data', function(){

			var m = mnistReader.zeroAndOne().map(function(datum){
				return [datum[0], datum[1].slice(0,2)]
			});
			
			var percentForTraining = 0.9;
			var trainingData = m.slice(0,Math.round(m.length*percentForTraining));
			var validationData = m.slice(Math.round(m.length*percentForTraining),m.length);
			
			var neuronOptions = {
				typeOfNeuron: 'tanh',
				randomness: 'flatProportionateZero',
				cost: 'squaredError'
			};

			var NN = new Network([
				{neuronOptions: neuronOptions, pattern: {dimensions: [trainingData[0][0].length]}},
				{neuronOptions: neuronOptions, pattern: {type: 'full', dimensions: [30]}},
				{neuronOptions: neuronOptions, pattern: {type: 'full', dimensions: [2]}}
			]);

			NN.init();
			console.log("Training...")
			for(var x = 0; x < trainingData.length / 8; x++){
				NN.activation(m[x][0])
				NN.propogate(m[x][1]);
				NN.adjust(0.0003);
			}
			console.log("Testing...");
			var good = 0;
			for(var x = 0; x < validationData.length; x++){
				var results = NN.activation(m[x][0]);
				if (m[x][1][0] == 1){
					if (results[0] > results[1]){
						good++;
					}
				}else{
					if (results[1] > results[0]){
						good++;
					}
				}
				//console.log("Results: ", results, "   Ideal: ", m[x][1])
			}
			console.log("Got ", good, " out of ", validationData.length, " right.")
			expect(good*1.2 > validationData.length).to.eql(true);

		});

		it('Can use a basic conv-net network to distinguish 1s and 0s MNIST data', function(){

			var m = mnistReader.zeroAndOne().map(function(datum){
				return [datum[0], datum[1].slice(0,2)]
			});
			
			var percentForTraining = 0.9;
			var trainingData = m.slice(0,Math.round(m.length*percentForTraining));
			var validationData = m.slice(Math.round(m.length*percentForTraining),m.length);
			
			var neuronOptions = {
				typeOfNeuron: 'tanh',
				randomness: 'flatProportionateZero',
				cost: 'squaredError'
			};

			var NN = new Network([
				{neuronOptions: neuronOptions, pattern: {dimensions: [Math.sqrt(trainingData[0][0].length), Math.sqrt(trainingData[0][0].length)]}},
				{neuronOptions: neuronOptions, pattern: {type: 'partial', stride:[1,1], field:[5,5], depth:[2]}},
				{neuronOptions: neuronOptions, pattern: {type: 'full', dimensions: [2]}}
			]);

			NN.init();
			console.log("Training...")
			for(var x = 0; x < trainingData.length / 8; x++){
				NN.activation(m[x][0])
				NN.propogate(m[x][1]);
				NN.adjust(0.0003);
			}
			console.log("Testing...");
			var good = 0;
			for(var x = 0; x < validationData.length; x++){
				var results = NN.activation(m[x][0]);
				if (m[x][1][0] == 1){
					if (results[0] > results[1]){
						good++;
					}
				}else{
					if (results[1] > results[0]){
						good++;
					}
				}
				//console.log("Results: ", results, "   Ideal: ", m[x][1])
			}
			console.log("Got ", good, " out of ", validationData.length, " right.")
			expect(good*1.2 > validationData.length).to.eql(true);

		});

		it('Can use a basic fully-connected network to distinguish ALL of the MNIST data', function(){

			var m = mnistReader.allElements()
			
			var percentForTraining = 0.9;
			var trainingData = m.slice(0,Math.round(m.length*percentForTraining));
			var validationData = m.slice(Math.round(m.length*percentForTraining),m.length);
			
			var neuronOptions = {
				typeOfNeuron: 'tanh',
				randomness: 'flatProportionateZero',
				cost: 'squaredError'
			};

			var NN = new Network([
				{neuronOptions: neuronOptions, pattern: {dimensions: [trainingData[0][0].length]}},
				{neuronOptions: neuronOptions, pattern: {type: 'full', dimensions: [30]}},
				{neuronOptions: neuronOptions, pattern: {type: 'full', dimensions: [10]}}
			]);

			NN.init();
			console.log("Training...")
			for(var x = 0; x < trainingData.length * 2; x++){
				NN.activation(m[x % trainingData.length][0])
				NN.propogate(m[x % trainingData.length][1]);
				NN.adjust(0.0002);
			}
			console.log("Testing...");
			var good = 0;
			for(var x = 0; x < validationData.length; x++){
				var results = NN.activation(m[x][0]);
				var maxIndex = results.indexOf(_.max(results))
				if (maxIndex == m[x][1].indexOf(1)){
					good++
				}
				//console.log("Results: ", results, "   Ideal: ", m[x][1])
			}
			console.log("Got ", good, " out of ", validationData.length, " right.")
			expect(good*2 > validationData.length).to.eql(true);


		});

		it('Can use a basic conv-net network to distinguish ALL of the MNIST data', function(){

		});

	});

});


