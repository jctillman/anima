var shared = {};

module.exports = function(neuronKind){
	if (neuronKind.indexOf('_') != -1){
		shared[neuronKind] = shared.hasOwnProperty(neuronKind) ? shared[neuronKind] : {};
		tempLink = shared[neuronKind];
		neuronKind = neuronKind.split('_')[0];
	}else{
		tempLink = {}
	}
	return tempLink;
	
}
