var _ = require('lodash');
var neuron = require('../neuron/neuron');
var latn = require('latn');

var Network = function(options){
	var self = this;
	self.layers = [];
	options.forEach(function(option, index){

		self.layers.push({ neurons: []});
		var thisLayer = self.layers[self.layers.length-1];
		if (index == 0){
			thisLayer.dimensions = option.pattern.dimensions;							//Repetition 1A--can abstract out
			latn.iter.apply({},option.pattern.dimensions.concat(function(value){
				thisLayer.neurons.push({
					neuron: neuron(option.neuronOptions),
					location: Array.prototype.slice.call(arguments)
				});
			}));
		}else{
			if (option.pattern.type == 'full'){
				thisLayer.dimensions = option.pattern.dimensions;						//Repetition 1A--can abstract out
				latn.iter.apply({},option.pattern.dimensions.concat(function(value){
					thisLayer.neurons.push({
						neuron: neuron(option.neuronOptions),
						location: Array.prototype.slice.call(arguments)
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
				});

				//Add neurons in dimensions for this layer.
				latn.iter.apply({},thisLayer.dimensions.concat(function(value){			//Reptition 1, without A.
					thisLayer.neurons.push({
						neuron: neuron(option.neuronOptions),
						location: Array.prototype.slice.call(arguments)
					});
				}));

				//For this layer, connect the right neurons to the right prior neurons.
				for(var x = 0; x < thisLayer.neurons.length; x++){
					//Get location, relative to the prior layer
					var relativeLocation = thisLayer.neurons[x].location.map(function(n, index){
						return n * option.pattern.stride[index];
					});
					//Loop over field, for this layer.

					latn.iter.apply({}, option.pattern.field.concat(function(){

						//Find location for neuron to which we need to connect.
						var fieldLocation = Array.prototype.slice.apply(arguments);
						var connectionLocation = fieldLocation.map(function(n,i){return relativeLocation[i]+n});

						var neuronToConnectTo = priorLayer.neurons.filter(function(neuron){
							return _.eq(neuron.location, connectionLocation)
						});
						if (neuronToConnectTo.length !== 1){throw new Error("!")}
						thisLayer.neurons[x].neuron.connect(neuronToConnectTo[0].neuron);

					}));
				}
			}
		}
	});
}

module.exports = Network
