module.exports = {
	'zero': function(){
		return 0;
	},
	'one': function(){
		return 1;
	},
	'flatPositive': function(){
		return Math.random();
	},
	'flatZero': function(){
		return (Math.random() - 0.5) * 1;
	},
	'flatProportionatePositive': function(n){
		return Math.random() / Math.sqrt(n)
	},
	'flatProportionateZero': function(n){
		return (Math.random() - 0.5) * 2 / Math.sqrt(n)
	}
}