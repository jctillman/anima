
var makeGenericTemplates = require('./makeGenericTemplates')

var shared = {};

module.exports = function(neuronKind, randomnessFunc, cost){

	generic = makeGenericTemplates(randomnessFunc, cost)
	var tempLink;

	if (neuronKind.indexOf('_') != -1){
		shared[neuronKind] = shared.hasOwnProperty(neuronKind) ? shared[neuronKind] : {};
		tempLink = shared[neuronKind];
		neuronKind = neuronKind.split('_')[0];
	}else{
		tempLink = {}
	}

	var baseTemplate = {
		connections: [],
		influences: [],
		'connect': generic.connect,
		'disconnect': generic.disconnect,
		'init': generic.init(tempLink),
		'propogate': generic.propogate,
		'adjust': generic.adjust,
		'applyDeltas': generic.applyDeltas,
		'activation': function(){throw new Error("This activation function needs to be replaced.")},
		'dAwrt': function(){throw new Error("This dAwrt function needs to be replaced.")},
		
		
	};

	templateExtensions = {

		'input': {
			'activation': generic.activation(function(z){
				throw new Error("Input neurons should always receive input, never calculate it.");
			}),
			'dAwrt': function(){
				throw new Error("Input neurons should never have anything before them!");
			},
			'propogate' : function(){
				throw new Error("There's no point to propogating an error to input neurons!");
			},
			'adjust': function(){
				throw new Error("There's no point in adjusting an input neuron!");
			}
		},

		'relu' : {
			'activation': generic.activation(function(z){
				return (z < 0) ? z / 20 : z;
			 }),
			'dAwrt': generic.dAwrt(function(self, neuronWeight, other){
				return (self.z) < 0 ? neuronWeight / 20 : neuronWeight;
			 }),
			'getDeltas': generic.getDeltas(function(self){
				return (self.z < 0) ? 0.05 : 1;
			})
		},

		'leakyrelu' : {
			'activation': generic.activation(function(z){
				return (z < 0) ? z / 5 : z;
			 }),
			'dAwrt': generic.dAwrt(function(self, neuronWeight, other){
				return self.z < 0 ? neuronWeight / 5 : neuronWeight;
			 }),
			'getDeltas': generic.getDeltas(function(self){
				return (self.z < 0) ? 0.2 : 1;
			})
		},

		'linear' : {
		 	'activation': generic.activation(function(z){
		 		return z
		 	 }),
		 	'dAwrt': generic.dAwrt(function(self, neuronWeight, other){
		 		return neuronWeight;
		 	 }),
			'getDeltas': generic.getDeltas(function(self){
				return 1;
			})
		},

		'tanh': {
			'activation': generic.activation(function(z){
				var eToX = Math.pow(Math.E, z);
				var eToNegX = Math.pow(Math.E, -z);
				return (eToX - eToNegX)/(eToX + eToNegX);
			 }),
			'dAwrt': generic.dAwrt(function(self, neuronWeight, other){
				var eTo2X = Math.pow(Math.E, 2*self.z);
				var eToNeg2X = Math.pow(Math.E, -2*self.z);
				var base = 4 / (eTo2X + 2 + eToNeg2X);
				return base * neuronWeight
			 }),

			 'getDeltas': generic.getDeltas(function(self){
				var eTo2X = Math.pow(Math.E, 2*self.z);
				var eToNeg2X = Math.pow(Math.E, -2*self.z);
				var dAwrtZ = 4 / (eTo2X + 2 + eToNeg2X);
				return dAwrtZ;
			 })

		},

		'sigmoid': {
			'activation': generic.activation(function(z){
				return 1 / (1 + Math.pow(Math.E, -z));
			 }),
			'dAwrt': generic.dAwrt(function(self, neuronWeight, other){
				var base = 1 / (Math.pow(Math.E, -self.z) + 2 + Math.pow(Math.E, self.z))
				return base * neuronWeight;
			 }),
			'getDeltas': generic.getDeltas(function(self){
				var dAwrtZ = 1 / (Math.pow(Math.E, -self.z) + 2 + Math.pow(Math.E, self.z));
				return dAwrtZ;
			 })
		},

	}

	for (var m in templateExtensions[neuronKind]) { baseTemplate[m] = templateExtensions[neuronKind][m]; }
	return baseTemplate

}




