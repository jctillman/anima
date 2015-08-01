module.exports = function(randomnessFunc, cost){

	return {

			'connect': function(neuron){
				this.connections.push(neuron);
				neuron.influences.push(this);
				return true;
			},

			'disconnect': function(neuron){
				if(this.connections.indexOf(neuron) != -1){
					neuron.influences.splice(neuron.influences.indexOf(this),1);
					this.connections.splice(this.connections.indexOf(neuron),1);
					return true;
				}
				return false;
			},

			'init' : function(){
				var self = this;
				
				//Initializing whatever is used in generic neurons.
				self.bias = randomnessFunc(self.connections.length);
				self.weights = [];
				self.connections.forEach(function(){
					self.weights.push(randomnessFunc(self.connections.length));
				});
				
				//This is used in stochastic gradient descent.
				self.numDeltas = 0;
				self.biasDelta = 0;
				self.weightsDelta = self.weights.map(function(){return 0;});

				//Initialization of activation value--start not following.
				self.a = 0;

				return true;
			},

			'propogate' : function(val){
				var self = this;
				if (!isNaN(val)){
					self.dCwrtA = cost.derivative(val, self.a);
					return self.dCwrtA;
				}else{

					self.dCwrtA = self.influences.reduce(function(sum, influencedNeuron, index){
						return sum + influencedNeuron.dCwrtA * influencedNeuron.dAwrt(self);
					},0);
					return self.dCwrtA;
				}
			},
			
			'adjust': function(eta){
				var self = this;
				self.getDeltas();
				self.applyDeltas(eta);
			},

			'getDeltas': function(calculateDawrtZ){
				return function(){
					var self = this;
					var dAwrtZ = calculateDawrtZ(self);
					
					self.biasDelta = self.biasDelta + ( self.dCwrtA * dAwrtZ);
					self.weightsDelta = self.weightsDelta.map(function(weightDelta, weightIndex){
						return weightDelta + (self.connections[weightIndex].a * self.dCwrtA * dAwrtZ);
					});
					self.numDeltas = self.numDeltas + 1;
					//console.log(self.weightsDelta)
					//console.log(self.numDeltas)
				}
			},

			'applyDeltas': function(eta){
				var self = this;
				var lambda = eta / self.numDeltas;
				
				//Apply deltas accumulated previously, while dividing by the num.
				self.bias = self.bias - (self.biasDelta * lambda);
				self.weights = self.weights.map(function(weight, weightIndex){
					return weight - (self.weightsDelta[weightIndex] * lambda)
				});

				//Get ready for the next deltas to be calculated.
				self.numDeltas = 0;
				self.biasDelta = 0;
				self.weightsDelta = self.weightsDelta.map(function(){return 0;});
			},

			'activation': function(link_function){
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
						self.a = link_function(self.z);
						return self.a;
					}
				}
			},

			'dAwrt': function(deriv_function){
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