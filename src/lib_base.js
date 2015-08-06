var generic = require('./lib_generic');

module.exports = function(obj){
	var base = {
		connections: [],
		influences: [],
		'connect': generic.connect_f,
		'disconnect': generic.disconnect_f,
		'adjust': generic.adjust_f,
		'applyDeltas': generic.applyDeltas_f
	};
	var keys = Object.keys(obj);
	for(var x = 0; x < keys.length; x++){
		base[keys[x]] = obj[keys[x]]
	}
	return base;
}