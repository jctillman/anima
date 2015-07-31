
var makeGenericTemplates = require('./makeGenericTemplates')

module.exports = function(neuronKind, randomnessFunc, cost){

	generic_templates = makeGenericTemplates(randomnessFunc, cost)

	var baseTemplate = {
		connections: [],
		influences: [],
		'connect': generic_templates.generic_connect,
		'disconnect': generic_templates.generic_disconnect,
		'init': generic_templates.generic_init,
		'propogate': generic_templates.generic_propogate,
		'adjust': generic_templates.generic_adjust,
		'activation': function(){throw new Error("This function needs to be replaced.")},
		'dAwrt': function(){throw new Error("This function needs to be replaced.")}
	};

	templateExtensions = {

		'input': {
			'activation': generic_templates.generic_activation(function(z){
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
			'activation': generic_templates.generic_activation(function(z){
				return (z < 0) ? z / 20 : z;
			 }),
			'dAwrt': generic_templates.generic_dAwrt(function(self, neuronWeight, other){
				return (self.z) < 0 ? neuronWeight / 20 : neuronWeight;
			 }),
			'adjust': function(eta){
				if (isNaN(eta)){throw new Error("Need to include learning rate in adjust.");}
				var self = this;
				var dAwrtZ = (self.z < 0) ? 1/20 : 1;
				self.bias = self.bias - ( self.dCwrtA * dAwrtZ * eta);
				self.weights = self.weights.map(function(weight, weightIndex){
					return weight - (self.connections[weightIndex].a * eta * self.dCwrtA * dAwrtZ);
				});
			}
		},

		'leakyrelu' : {
			'activation': generic_templates.generic_activation(function(z){
				return (z < 0) ? z / 5 : z;
			 }),
			'dAwrt': generic_templates.generic_dAwrt(function(self, neuronWeight, other){
				return self.z < 0 ? neuronWeight / 5 : neuronWeight;
			 }),
			'adjust': function(eta){
				if (isNaN(eta)){throw new Error("Need to include learning rate in adjust.");}
				var self = this;
				var dAwrtZ = (self.z < 0) ? 0.2 : 1;
				self.bias = self.bias - ( self.dCwrtA * dAwrtZ * eta);
				self.weights = self.weights.map(function(weight, weightIndex){
					return weight - (self.connections[weightIndex].a * eta * self.dCwrtA * dAwrtZ);
				});
			}
		},

		'linear' : {
		 	'activation': generic_templates.generic_activation(function(z){
		 		return z
		 	 }),
		 	'dAwrt': generic_templates.generic_dAwrt(function(self, neuronWeight, other){
		 		return neuronWeight;
		 	 }),
		 	'adjust': function(eta){
				if (isNaN(eta)){throw new Error("Need to include learning rate in adjust.");}
				var self = this;
				var dAwrtZ = 1;
				self.bias = self.bias - ( self.dCwrtA * dAwrtZ * eta);
				self.weights = self.weights.map(function(weight, weightIndex){
					return weight - (self.connections[weightIndex].a * eta * self.dCwrtA * dAwrtZ);
			
				});
			}
		},

		'tanh': {
			'activation': generic_templates.generic_activation(function(z){
				var eToX = Math.pow(Math.E, z);
				var eToNegX = Math.pow(Math.E, -z);
				return (eToX - eToNegX)/(eToX + eToNegX);
			 }),
			'dAwrt': generic_templates.generic_dAwrt(function(self, neuronWeight, other){
				var eTo2X = Math.pow(Math.E, 2*self.z);
				var eToNeg2X = Math.pow(Math.E, -2*self.z);
				var base = 4 / (eTo2X + 2 + eToNeg2X);
				return base * neuronWeight
			 }),
			'adjust': function(eta){
				if (isNaN(eta)){throw new Error("Need to include learning rate in adjust.");}
				var self = this;
				var eTo2X = Math.pow(Math.E, 2*self.z);
				var eToNeg2X = Math.pow(Math.E, -2*self.z);
				var dAwrtZ = 4 / (eTo2X + 2 + eToNeg2X);
				self.bias = self.bias - ( self.dCwrtA * dAwrtZ * eta);
				self.weights = self.weights.map(function(weight, weightIndex){
					return weight - (self.connections[weightIndex].a * eta * self.dCwrtA * dAwrtZ);
				});
			}

		},

		'sigmoid': {
			'activation': generic_templates.generic_activation(function(z){
				return 1 / (1 + Math.pow(Math.E, -z));
			 }),
			'dAwrt': generic_templates.generic_dAwrt(function(self, neuronWeight, other){
				var base = 1 / (Math.pow(Math.E, -self.z) + 2 + Math.pow(Math.E, self.z))
				return base * neuronWeight;
			 }),
			'adjust': function(eta){
				if (isNaN(eta)){throw new Error("Need to include learning rate in adjust.");}
				var self = this;
				var dAwrtZ = 1 / (Math.pow(Math.E, -self.z) + 2 + Math.pow(Math.E, self.z))
				self.bias = self.bias - ( self.dCwrtA * dAwrtZ * eta);
				self.weights = self.weights.map(function(weight, weightIndex){
					return weight - (self.connections[weightIndex].a * self.dCwrtA * dAwrtZ * eta);
				});
			}
		},

	}


	for (var m in templateExtensions[neuronKind]) { baseTemplate[m] = templateExtensions[neuronKind][m]; }
	return baseTemplate

}




