var shared = {};

module.exports = function(neuronKind){
	if (neuronKind.indexOf('_') != -1){
		neuronKind = neuronKind.split('_')[1];
		//console.log(shared)
		shared[neuronKind] = shared.hasOwnProperty(neuronKind) ? shared[neuronKind] : {uniqueId: Math.floor(Math.random()*100)};
		tempLink = shared[neuronKind];
		
	}else{
		tempLink = {}
	}
	return tempLink;
	
}
