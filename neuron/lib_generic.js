module.exports =  {

			'connect_f': function(neuron){
				this.connections.push(neuron);
				neuron.influences.push(this);
				return true;
			},

			'disconnect_f': function(neuron){
				if(this.connections.indexOf(neuron) != -1){
					neuron.influences.splice(neuron.influences.indexOf(this),1);
					this.connections.splice(this.connections.indexOf(neuron),1);
					return true;
				}
				return false;
			},

			'init_fm' : function(sharedValue, randomnessFunc){

				var sharedValue = sharedValue;

				return function(){
					var self = this;
					
					//Initializing parameters for the neurons.
					//console.log("before", !self.connections || self.connections.length, !sharedValue.weights || sharedValue.weights.length)
					if(!sharedValue.hasOwnProperty('weights')){
						sharedValue.bias = randomnessFunc(self.connections.length);
						sharedValue.weights = [];
						self.connections.forEach(function(){
							sharedValue.weights.push(randomnessFunc(self.connections.length));
						});
					}
					//console.log("after", self.connections.length, sharedValue.weights.length)
					self.sv = sharedValue;
					
					//This is used in stochastic gradient descent.
					self.numDeltas = 0;
					self.biasDelta = 0;
					self.weightsDelta = self.sv.weights.map(function(){return 0;});

					//Initialization of activation value--start not following.
					self.a = 0;

					return true;
				}
			},

			'propogate_fm' : function(cost){
				return function(val){
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
				}
			},
			
			'adjust_f': function(eta){
				var self = this;
				self.getDeltas();
				self.applyDeltas(eta);
			},

			'getDeltas_fm': function(calculateDawrtZ){
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

			'applyDeltas_f': function(eta){
				var self = this;
				var lambda = eta / self.numDeltas;
				
				//Apply deltas accumulated previously, while dividing by the num.
				self.sv.bias = self.sv.bias - (self.biasDelta * lambda);
				self.sv.weights = self.sv.weights.map(function(weight, weightIndex){
					return weight - (self.weightsDelta[weightIndex] * lambda)
				});

				//Get ready for the next deltas to be calculated.
				self.numDeltas = 0;
				self.biasDelta = 0;
				self.weightsDelta = self.weightsDelta.map(function(){return 0;});
			},

			'activation_fm': function(link_function){
				return function(val){
					var self = this;
					if (!isNaN(val)){
						self.a = val;
						return self.a;
					}
					else{
						//console.log(self.sv.weights, self.connections.length);
						self.z = self.connections.reduce(function(sum, neuron, index){
							//console.log("degenerate", sum, neuron.a, self.sv.weights[index]); //So the problem is weights has nine elements, but connections apparently has more?
							return sum + neuron.a * self.sv.weights[index]
						}, self.sv.bias);
						self.a = link_function(self.z);
						//console.log("a", self.a)
						return self.a;
					}
				}
			},

			'dAwrt_fm': function(deriv_function){
				return function(neuron){
					var self = this;
					var index = self.connections.indexOf(neuron);
					if (index == -1){
						return 0
					}else{
						return deriv_function(self, self.sv.weights[index], neuron);
					}
				}
			}
		}
