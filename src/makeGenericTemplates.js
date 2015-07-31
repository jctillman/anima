module.exports = function(randomnessFunc, cost){

	return {

			'generic_connect': function(neuron){
				this.connections.push(neuron);
				neuron.influences.push(this);
				return true;
			},

			'generic_disconnect': function(neuron){
				if(this.connections.indexOf(neuron) != -1){
					neuron.influences.splice(neuron.influences.indexOf(this),1);
					this.connections.splice(this.connections.indexOf(neuron),1);
					return true;
				}
				return false;
			},

			'generic_init' : function(){
				var self = this;
				self.weights = [];
				self.connections.forEach(function(){
					self.weights.push(randomnessFunc(self.connections.length));
				});
				self.bias = randomnessFunc(self.connections.length);
				self.a = 0;
				return true;
			},

			'generic_propogate' : function(val){
				var self = this;
				if (!isNaN(val)){
					//console.log("?", self.a, cost.derivative(val, self.a));
					self.dCwrtA = cost.derivative(val, self.a);
					//console.log("ASD", val, self.a, self.dCwrtA)
					//consol
					return self.dCwrtA;
				}else{

					self.dCwrtA = self.influences.reduce(function(sum, influencedNeuron, index){
						//console.log("sd", sum, influencedNeuron.dCwrtA, influencedNeuron.dAwrt(self))
						return sum + influencedNeuron.dCwrtA * influencedNeuron.dAwrt(self);
					},0);
					return self.dCwrtA;
				}
			},
			//Fuck, this is all wrong.
			'generic_adjust': function(eta){
				if (isNaN(eta)){throw new Error("Need to include learning rate in adjust.");}
				var self = this;
				self.bias = self.bias - self.dCwrtA * eta;
				self.bias = self.bias * 0.995;
				self.weights = self.weights.map(function(weight, weightIndex){
					var m = weight - (self.connections[weightIndex].a * eta * self.dCwrtA);
					return m * 0.995;
				});
			},

			'generic_activation': function(link_function){
				return function(val){
					var self = this;
					if (!isNaN(val)){
						self.a = val;
						return self.a;
					}
					else{
						self.z = self.connections.reduce(function(sum, neuron, index){
							return sum + neuron.a * self.weights[index]
						}, self.bias);
						//if(isNaN(self.z)){
						//	console.log(self)
						//	throw new Error(self);
						//}
						self.a = link_function(self.z);
						return self.a;
					}
				}
			},

			'generic_dAwrt': function(deriv_function){
				return function(neuron){
					var self = this;
					var index = self.connections.indexOf(neuron);
					if (index == -1){
						return 0
					}else{
						return deriv_function(self, self.weights[index], neuron);
					}
				}
			}
		}

}