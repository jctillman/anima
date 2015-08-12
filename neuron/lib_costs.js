module.exports = {
	'linearError': {
		'value': function(target, actual){
			return Math.abs(actual - target)
		},
		'derivative': function(target, actual){
			return (actual - target) < 0 ? -1 : 1
		}
	},
	'squaredError': {
		'value': function(target, actual){
			return 0.5 * Math.pow(actual - target, 2)
		},
		'derivative': function(target, actual){
			return (actual - target);
		}

	},
	'hypercubedError': {
		'value': function(target, actual){
			return 0.25 * Math.pow(actual - target, 4)
		},
		'derivative': function(target, actual){
			return Math.pow(actual - target,3);
		}
	}
}