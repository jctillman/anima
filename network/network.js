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

		//Create the layer
		self.layers.push({ neurons: [], uniqueId: Math.floor(Math.random()*10000) });

		//Shorten, because otherwise things get unintelligible. lyr = this layer, prr = prior layer.
		var lyr = self.layers[self.layers.length-1];
		var prr = (self.layers.length > 1) ? self.layers[self.layers.length-2] : undefined;

		//The dimensions of this layer are either
		//(1) what the options say or
		//(2) determined from stride and field in options, and dimensions of the prior layer.
		lyr.dimensions =
			option.pattern.dimensions
			||
			option.pattern.stride.map(function(stride,index){
				return (prr.dimensions[index] - option.pattern.field[index])/stride + 1;
			}).concat( (option.pattern.depth != undefined) ? (option.pattern.depth) : [] );

		//Fill the determined dimensions with neurons.
		//If it has a depth property, we're dealing with something with layers.
		fillDimensions(lyr, lyr.dimensions, option, (!!option.pattern.depth) )
		
		//Connect the neurons to the neurons o the prior layer,
		//if the neurons have a connection type of'full' or 'partial' but not 'none'.
		for(var x = 0; x < lyr.neurons.length; x++){
			if (option.pattern.type == 'full'){
				for(var y = 0; y < prr.neurons.length; y++){
					lyr.neurons[x].neuron.connect(prr.neurons[y].neuron);
				}
			}else if (option.pattern.type == 'partial'){
				var relativeLocation = lyr.neurons[x].location.map(function(n, index){return n * ( option.pattern.stride[index] || 1);});
				latn.iter.apply({}, option.pattern.field.concat(function(){
					var connectionLocation = Array.prototype.slice.apply(arguments).map(function(n,i){return relativeLocation[i]+n});
					var neuronToConnectTo = prr.neurons.filter(function(neuron){
						return neuron.location.every(function(vr, ind){
							return vr == connectionLocation[ind];
						});
					});
					lyr.neurons[x].neuron.connect(neuronToConnectTo[0].neuron);
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

Network.prototype.batchTrain = function(inputs, outputs, eta, batchSize){
	for(var x = 0; x < inputs.length; x++){
		this.activation(inputs[x]);
		this.propogate(outputs[x]);
		this.getDeltas();
		((x + 1) % batchSize) || this.applyDeltas(eta);
	}
}

//Assumes we're dealing with one-hot encoding. 
Network.prototype.percentRight = function(inputs, outputs){
	var good = 0;
	for(var x = 0; x < inputs.length; x++){
		var results = this.activation(inputs[x]);
		var maxIndex = results.indexOf(_.max(results))
		if (maxIndex == outputs[x].indexOf(1)){
			good++
		}
	}
	return good / inputs.length;
}

module.exports = Network
