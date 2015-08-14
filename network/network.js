var _ = require('lodash');
var neuron = require('../neuron/neuron');
var latn = require('latn');

var Network = function(options){
	var self = this;
	self.layers = [];
	options.forEach(function(option, index){

		self.layers.push({ neurons: [], uniqueId: Math.floor(Math.random()*10000) });
		var thisLayer = self.layers[self.layers.length-1];
		if (index == 0){
			thisLayer.dimensions = option.pattern.dimensions;							//Repetition 1A--can abstract out
			latn.iter.apply({},option.pattern.dimensions.concat(function(value){
				var location = Array.prototype.slice.call(arguments)
				thisLayer.neurons.push({
					neuron: neuron(option.neuronOptions),
					location: location
				});
			}));
		}else{
			if (option.pattern.type == 'full'){
				thisLayer.dimensions = option.pattern.dimensions;						//Repetition 1A--can abstract out
				latn.iter.apply({},option.pattern.dimensions.concat(function(value){
					var location = Array.prototype.slice.call(arguments)
					thisLayer.neurons.push({
						neuron: neuron(option.neuronOptions),
						location: location
					});
				}));
				var priorLayer = self.layers[self.layers.length-2];
				for(var x = 0; x < thisLayer.neurons.length; x++){
					for(var y = 0; y < priorLayer.neurons.length; y++){
						thisLayer.neurons[x].neuron.connect(priorLayer.neurons[y].neuron);
					}
				}
			}else if (option.pattern.type == 'partial'){

				//Determine dimensions for this layer, from stride and field
				var priorLayer = self.layers[self.layers.length-2];
				thisLayer.dimensions = option.pattern.stride.map(function(stride,index){
					return (priorLayer.dimensions[index] - option.pattern.field[index])/stride + 1;
				}).concat( (option.pattern.depth != undefined) ? (option.pattern.depth) : [] );

				//Add neurons in dimensions for this layer.
				if(option.pattern.depth == undefined){
					//console.log("d", thisLayer.dimensions)
					latn.iter.apply({},thisLayer.dimensions.concat(function(value){			//Reptition 1, without A.
						var location  = Array.prototype.slice.call(arguments)
						//console.log("loc", location)
						thisLayer.neurons.push({
							neuron: neuron(option.neuronOptions),
							location: Array.prototype.slice.call(arguments)
						});
					}));
				}else{
					//console.log("!!!!")
					var free = _.cloneDeep(option.neuronOptions);
					var base = free.typeOfNeuron;
					//console.log("d", thisLayer.dimensions)
					latn.iter.apply({},thisLayer.dimensions.concat(function(value){			//Reptition 1, without A.
						var location  = Array.prototype.slice.call(arguments)
						//console.log("loc", location, thisLayer.uniqueId)
						free.typeOfNeuron = base + "_" + location[location.length-1] + "" + thisLayer.uniqueId;
						//console.log(free.typeOfNeuron);
						thisLayer.neurons.push({
							neuron: neuron(free),
							location: location
						});
					}));
				}

				//For this layer, connect the right neurons to the right prior neurons.
				for(var x = 0; x < thisLayer.neurons.length; x++){
					//Get location, relative to the prior layer
					var relativeLocation = thisLayer.neurons[x].location.map(function(n, index){
						return n * ( option.pattern.stride[index] || 1);
					});
					//console.log("rel", relativeLocation)
					//Loop over field, for this layer.

					latn.iter.apply({}, option.pattern.field.concat(function(){

						//Find location for neuron to which we need to connect.
						var fieldLocation = Array.prototype.slice.apply(arguments);
						//console.log("field", fieldLocation)
						var connectionLocation = fieldLocation.map(function(n,i){return relativeLocation[i]+n});
						//console.log('connecting neuron at ' + thisLayer.neurons[x].location + ' to ' + connectionLocation)// fieldLocation)

						var neuronToConnectTo = priorLayer.neurons.filter(function(neuron){
							return neuron.location.every(function(vr, ind){
								return vr == connectionLocation[ind];
							});
						});
						if (neuronToConnectTo.length !== 1){
							//console.log(neuronToConnectTo.length);
							throw new Error("!");
						}

						thisLayer.neurons[x].neuron.connect(neuronToConnectTo[0].neuron);

					}));
				}
			}
		}
	});
}

Network.prototype.init = function(){
	var self = this;
	for(var x = 0; x < self.layers.length; x++){
		for(var y = 0; y < self.layers[x].neurons.length; y++){
			self.layers[x].neurons[y].neuron.init();
		}
	}
}

Network.prototype.activation = function(val){
	var self = this;
	for(var x = 0; x < self.layers[0].neurons.length; x++){

		self.layers[0].neurons[x].neuron.activation(val[x]);
	}
	for(var x = 1; x < self.layers.length; x++){
		for(var y = 0; y < self.layers[x].neurons.length; y++){
			
			self.layers[x].neurons[y].neuron.activation();
			//console.log(self.layers[x].neurons[y].neuron)
		}
	}
	//console.log(self.layers[self.layers.length-1].neurons)
	return self.layers[self.layers.length-1].neurons.map(function(n){
		return n.neuron.a;
	});
}

Network.prototype.propogate = function(val){
	var self = this;
	var last = self.layers[self.layers.length-1].neurons;
	for(var x = 0; x < last.length; x++){
		last[x].neuron.propogate(val[x]);
	}
	for(var x = self.layers.length-2; x > 0; x--){
		for(var y = 0; y < self.layers[x].neurons.length; y++){
			self.layers[x].neurons[y].neuron.propogate();
		}	
	}
}

Network.prototype.adjust = function(eta){

	var self = this;
	var last = self.layers[self.layers.length-1].neurons;
	for(var x = 0; x < last.length; x++){
		last[x].neuron.adjust(eta);
	}
	for(var x = self.layers.length-2; x > 0; x--){
		for(var y = 0; y < self.layers[x].neurons.length; y++){
			self.layers[x].neurons[y].neuron.adjust(eta);
		}	
	}

}

Network.prototype.applyDeltas = function(eta){

	var self = this;
	var last = self.layers[self.layers.length-1].neurons;
	for(var x = 0; x < last.length; x++){
		last[x].neuron.applyDeltas(eta);
	}
	for(var x = self.layers.length-2; x > 0; x--){
		for(var y = 0; y < self.layers[x].neurons.length; y++){
			self.layers[x].neurons[y].neuron.applyDeltas(eta);
		}	
	}

}

Network.prototype.getDeltas = function(){

	var self = this;
	var last = self.layers[self.layers.length-1].neurons;
	for(var x = 0; x < last.length; x++){
		last[x].neuron.getDeltas();
	}
	for(var x = self.layers.length-2; x > 0; x--){
		for(var y = 0; y < self.layers[x].neurons.length; y++){
			self.layers[x].neurons[y].neuron.getDeltas();
		}	
	}

}

module.exports = Network
