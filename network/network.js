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

		//Fill this layer with neurons.
		//The dimensions of this layer are either (1) what it says or (2) determined from stride, field, and dimensions of prior layer.
		thisLayer.dimensions = option.pattern.dimensions || option.pattern.stride.map(function(stride,index){
				return (self.layers[self.layers.length-2].dimensions[index] - option.pattern.field[index])/stride + 1;
			}).concat( (option.pattern.depth != undefined) ? (option.pattern.depth) : [] );
		fillDimensions(thisLayer, thisLayer.dimensions, option, (!!option.pattern.depth) )
		
		//Connect the neurons 
		for(var x = 0; x < thisLayer.neurons.length; x++){

			if (option.pattern.type == 'full'){
				for(var y = 0; y < self.layers[self.layers.length-2].neurons.length; y++){
					thisLayer.neurons[x].neuron.connect(self.layers[self.layers.length-2].neurons[y].neuron);
				}

			}else if (option.pattern.type == 'partial'){
				//Get location, relative to the prior layer
				var relativeLocation = thisLayer.neurons[x].location.map(function(n, index){return n * ( option.pattern.stride[index] || 1);});
				//Loop over field, for this layer.
				latn.iter.apply({}, option.pattern.field.concat(function(){
					var connectionLocation = Array.prototype.slice.apply(arguments).map(function(n,i){return relativeLocation[i]+n});
					var neuronToConnectTo = self.layers[self.layers.length-2].neurons.filter(function(neuron){
						return neuron.location.every(function(vr, ind){
							return vr == connectionLocation[ind];
						});
					});
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
