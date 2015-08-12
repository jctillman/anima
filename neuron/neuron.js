var randomnesses = require('./lib_randomnesses');
var costs = require('./lib_costs');
var sharedMemory = require('./lib_shared');
var shared = {};

var Neuron = function(options){
		
	var neuronKind = options.typeOfNeuron || 'leakyrelu';
	var randomness = options.randomness || 'flatProportionateZero';
	var cost = options.cost || 'squaredError';

	var rand = randomnesses[randomness]
	var cost = costs[cost];
	var link = sharedMemory(neuronKind);
	var neuk = neuronKind.split('_')[0];
	
	return require('./lib_kinds')[neuk](rand, cost, link);
	
}

module.exports = Neuron

