var neurons = require('./neurons');
var randomnesses = require('./randomnesses')
var costs = require('./costs');


var Neuron = function(options){
		
	var typeOfNeuron = options.typeOfNeuron || 'relu';
	var randomness = options.randomness || 'flatProportionatePositive';
	var cost = options.cost || 'squaredError';

	var rand = randomnesses[randomness]
	var cost = costs[cost];
	template = neurons(typeOfNeuron, rand, cost);


	return template;
}

module.exports = Neuron