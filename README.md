#Anima

##A JS neural network

*In progress.  Backwards compatibility not guaranteed.*

Anima lets you train feedforward neural networks, with or without convolutional layers.  It consists of a neuron class, which has most of the functionnality, and a network class, which provides useful functions for iterating over layers consisting of instances of the neuron class.  The below documentation is suggestive rather than exaustive, tentative, and very possibly temporary.

## Neuron

> var neuronMaker = require('./neuron');
>
> var inputNeuron = neuron({
>
> 	typeOfNeuron: 'input'
>
> });
>
> var neuronInstance = neuron({
>
> 	typeOfNeuron: 'leakyrelu',
>
>	randomness: 'flatProportionateZero'
>
> });
>
> var secondNeuron = neuron({
>
> 	typeOfNeuron: 'leakyrelu',
>
>	cost: 'squaredError'
>
> })
>
> //Hook them up and initialize them
>
> neuronInstance.connect(inputNeuron);
>
> secondNeuron.connect(neuronInstance);
> inputNeuron.init();
>
> neuronInstance.init();
>
> secondNeuron.init();
>
> //Set input for neuron
>
> inputNeuron.activation(1);
>
> neuronInstance.activate();
>
> secondNeuron.activate();
>
> //Set what the neuron should be and propogate error back
>
> secondNeuron.propogate(1)
>
> neuronInstance.propogate();
>
> //Adjust
>
> secondNeuron.adjust(0.1);
>
> neuronInstance.adjust(0.1);
