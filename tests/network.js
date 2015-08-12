var Network = require('../network/network');
var expect = require('chai').expect;

describe('Testing network, that it creates networks and feeds forward basically right', function(){

	describe('It can make a fully-connected network, with as many layers as specified, of the dimensions specified', function(){

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

	describe('It can make a partially connected network without shared weights, with as many layers as specified, of the dimensions specified', function(){

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

	});

	describe('It can make a partly-connected network with shared weights, uh-huh, uh-huh', function(){

		it('Can make a partly-connected network with shared weights, moving from a 1d to a 2d field', function(){

		});

		it('Can make a partly-connected network with shared weights, moving from a 2d to a 3d field', function(){

		});

		it('Can make a partly-connected network with shared weights, moving from a 3d to a 3d field', function(){

		});

	});

});