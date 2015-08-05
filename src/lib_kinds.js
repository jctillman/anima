var generic = require('./lib_generic');

module.exports = {

		'input': function(randomnessFunc, cost, tempLink){
			return {
				connections: [],
				influences: [],
				'connect': generic.connect_f,
				'disconnect': generic.disconnect_f,
				'init': generic.init_fm(tempLink, randomnessFunc),
				'propogate': generic.propogate_fm(cost),
				'adjust': generic.adjust_f,
				'applyDeltas': generic.applyDeltas_f,
				'activation': generic.activation_fm(function(z){
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
			}
		},

		'relu' : function(randomnessFunc, cost, tempLink){
			return {
				connections: [],
				influences: [],
				'connect': generic.connect_f,
				'disconnect': generic.disconnect_f,
				'init': generic.init_fm(tempLink, randomnessFunc),
				'propogate': generic.propogate_fm(cost),
				'adjust': generic.adjust_f,
				'applyDeltas': generic.applyDeltas_f,

				'activation': generic.activation_fm(function(z){
					return (z < 0) ? z / 20 : z;
				 }),
				'dAwrt': generic.dAwrt_fm(function(self, neuronWeight, other){
					return (self.z) < 0 ? neuronWeight / 20 : neuronWeight;
				 }),
				'getDeltas': generic.getDeltas_fm(function(self){
					return (self.z < 0) ? 0.05 : 1;
				})
			}
		},

		'leakyrelu' : function(randomnessFunc, cost, tempLink){
			return {
				connections: [],
				influences: [],
				'connect': generic.connect_f,
				'disconnect': generic.disconnect_f,
				'init': generic.init_fm(tempLink, randomnessFunc),
				'propogate': generic.propogate_fm(cost),
				'adjust': generic.adjust_f,
				'applyDeltas': generic.applyDeltas_f,
				'activation': generic.activation_fm(function(z){
					return (z < 0) ? z / 5 : z;
				 }),
				'dAwrt': generic.dAwrt_fm(function(self, neuronWeight, other){
					return self.z < 0 ? neuronWeight / 5 : neuronWeight;
				 }),
				'getDeltas': generic.getDeltas_fm(function(self){
					return (self.z < 0) ? 0.2 : 1;
				})
			}
		},

		'linear' : function(randomnessFunc, cost, tempLink){
			return {
				connections: [],
				influences: [],
				'connect': generic.connect_f,
				'disconnect': generic.disconnect_f,
				'init': generic.init_fm(tempLink, randomnessFunc),
				'propogate': generic.propogate_fm(cost),
				'adjust': generic.adjust_f,
				'applyDeltas': generic.applyDeltas_f,
			 	'activation': generic.activation_fm(function(z){
			 		return z
			 	 }),
			 	'dAwrt': generic.dAwrt_fm(function(self, neuronWeight, other){
			 		return neuronWeight;
			 	 }),
				'getDeltas': generic.getDeltas_fm(function(self){
					return 1;
				})
			}
		},

		'tanh': function(randomnessFunc, cost, tempLink){
			return {
				connections: [],
				influences: [],
				'connect': generic.connect_f,
				'disconnect': generic.disconnect_f,
				'init': generic.init_fm(tempLink, randomnessFunc),
				'propogate': generic.propogate_fm(cost),
				'adjust': generic.adjust_f,
				'applyDeltas': generic.applyDeltas_f,
				'activation': generic.activation_fm(function(z){
					var eToX = Math.pow(Math.E, z);
					var eToNegX = Math.pow(Math.E, -z);
					return (eToX - eToNegX)/(eToX + eToNegX);
				 }),
				'dAwrt': generic.dAwrt_fm(function(self, neuronWeight, other){
					var eTo2X = Math.pow(Math.E, 2*self.z);
					var eToNeg2X = Math.pow(Math.E, -2*self.z);
					var base = 4 / (eTo2X + 2 + eToNeg2X);
					return base * neuronWeight
				 }),

				 'getDeltas': generic.getDeltas_fm(function(self){
					var eTo2X = Math.pow(Math.E, 2*self.z);
					var eToNeg2X = Math.pow(Math.E, -2*self.z);
					var dAwrtZ = 4 / (eTo2X + 2 + eToNeg2X);
					return dAwrtZ;
				 })
			}

		},

		'sigmoid': function(randomnessFunc, cost, tempLink){
			return {
				connections: [],
				influences: [],
				'connect': generic.connect_f,
				'disconnect': generic.disconnect_f,
				'init': generic.init_fm(tempLink, randomnessFunc),
				'propogate': generic.propogate_fm(cost),
				'adjust': generic.adjust_f,
				'applyDeltas': generic.applyDeltas_f,
				'activation': generic.activation_fm(function(z){
					return 1 / (1 + Math.pow(Math.E, -z));
				 }),
				'dAwrt': generic.dAwrt_fm(function(self, neuronWeight, other){
					var base = 1 / (Math.pow(Math.E, -self.z) + 2 + Math.pow(Math.E, self.z))
					return base * neuronWeight;
				 }),
				'getDeltas': generic.getDeltas_fm(function(self){
					var dAwrtZ = 1 / (Math.pow(Math.E, -self.z) + 2 + Math.pow(Math.E, self.z));
					return dAwrtZ;
				 })
			}
		}

	};