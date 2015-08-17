var _ = require('lodash');
var neuron = require('../neuron/neuron');
var latn = require('latn');

var fillDimensions = function(aLayer, dimensions, options, usesConvolutions){
	var free = _.cloneDeep(options.neuronOptions);
	var base = free.typeOfNeuron;
	aLayer.dimensions = dimensions;
	latn.iter.apply({},dimensions.concat(function(value){
			var location = Array.prototype.slice.call(arguments)
			free.typeOfNeuron = (!usesConvolutions) ? base : base + "_" + location[location.length-1] + "" + aLayer.uniqueId;
			aLayer.neurons.push({
				neuron: neuron(free),
				location: location
		});
	}));
}

var Network = function(options){
	var self = this;
	self.layers = [];

	options.forEach(function(option, index){

		self.layers.push({ neurons: [], uniqueId: Math.floor(Math.random()*10000) });
		var thisLayer = self.layers[self.layers.length-1];

		//If we are at the first layer, just fill up the dimensions.
		if (index == 0){
			fillDimensions(thisLayer, option.pattern.dimensions, option)

		//If we are in a fully connected layer, fill up the dimensions and connect it to all prior.
		}else if (option.pattern.type == 'full'){
			fillDimensions(thisLayer, option.pattern.dimensions, option)
			var priorLayer = self.layers[self.layers.length-2];
			for(var x = 0; x < thisLayer.neurons.length; x++){
				for(var y = 0; y < priorLayer.neurons.length; y++){
					thisLayer.neurons[x].neuron.connect(priorLayer.neurons[y].neuron);
				}
			}

		//If we are in a partially connected layer, determine the dimensions,
		//make the neurons, and connect them.
		}else if (option.pattern.type == 'partial'){

			//Determine dimensions for this layer, from stride and field
			var priorLayer = self.layers[self.layers.length-2];
			thisLayer.dimensions = option.pattern.stride.map(function(stride,index){
				return (priorLayer.dimensions[index] - option.pattern.field[index])/stride + 1;
			}).concat( (option.pattern.depth != undefined) ? (option.pattern.depth) : [] );

			//Add neurons for the dimensions for this layer.
			if(option.pattern.depth == undefined){
				fillDimensions(thisLayer, thisLayer.dimensions, option)
			}else{
				fillDimensions(thisLayer, thisLayer.dimensions, option, true)
				// var free = _.cloneDeep(option.neuronOptions);
				// var base = free.typeOfNeuron;
				// latn.iter.apply({},thisLayer.dimensions.concat(function(value){			//Reptition 1, without A.
				// 	var location  = Array.prototype.slice.call(arguments)
				// 	//free.typeOfNeuron = base + "_" + location[location.length-1] + "" + thisLayer.uniqueId;
				// 	thisLayer.neurons.push({
				// 		neuron: neuron(free),
				// 		location: location
				// 	});
				// }));

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
					var connectionLocation = fieldLocation.map(function(n,i){return relativeLocation[i]+n});

					var neuronToConnectTo = priorLayer.neurons.filter(function(neuron){
						return neuron.location.every(function(vr, ind){
							return vr == connectionLocation[ind];
						});
					});
					if (neuronToConnectTo.length !== 1){
						throw new Error("!");
					}

					thisLayer.neurons[x].neuron.connect(neuronToConnectTo[0].neuron);

				}));
			}
		}
	});
}

Network.prototype.init = function(){
	this._forwards('init');
}

Network.prototype.activation = function(val){
	return this._forwards('activation', val);
}

Network.prototype.propogate = function(val){
	this._backwards('propogate', val);
}

Network.prototype.adjust = function(eta){
	this._backwards('adjust', eta);
}

Network.prototype.applyDeltas = function(eta){
	this._backwards('applyDeltas', eta);
}

Network.prototype.getDeltas = function(){
	this._backwards('getDeltas');
}

Network.prototype._forwards = function(funcName, value){

	var self = this;
	if (Array.isArray(value)){
		for(var x = 0; x < self.layers[0].neurons.length; x++){
			self.layers[0].neurons[x].neuron[funcName](value[x]);
		}
	}

	for(var x = 1; x < self.layers.length; x++){
		for(var y = 0; y < self.layers[x].neurons.length; y++){
			self.layers[x].neurons[y].neuron[funcName]();
		}
	}

	return self.layers[self.layers.length-1].neurons.map(function(n){
		return n.neuron.a;
	});

}

Network.prototype._backwards = function(funcName, value){

	var self = this;
	var last = self.layers[self.layers.length-1].neurons;
	if(!isNaN(value)){
		for(var x = 0; x < last.length; x++){
			last[x].neuron[funcName](value);
		}
	}else if(Array.isArray(value)){
		for(var x = 0; x < last.length; x++){
			last[x].neuron[funcName](value[x]);
		}
	}else{
		for(var x = 0; x < last.length; x++){
			last[x].neuron[funcName]();
		}
	}
	for(var x = self.layers.length-2; x > 0; x--){
		for(var y = 0; y < self.layers[x].neurons.length; y++){
			self.layers[x].neurons[y].neuron[funcName](value);
		}	
	}
};

module.exports = Network
