var neuron = require('../src/neuron');
var expect = require('chai').expect;

describe('Testing neuron, which is the neuron-creating function', function(){

	var allNeuronKinds = ['linear','leakyrelu','relu','tanh', 'sigmoid'];

	it('tests simple connection / diconnection items for all kinds', function(){

		for(var x = 0 ; x < allNeuronKinds.length; x++){
			var m = neuron({typeOfNeuron: allNeuronKinds[x] });
			var n = neuron({typeOfNeuron: allNeuronKinds[x] });
			var o = neuron({typeOfNeuron: allNeuronKinds[x] });
			expect(m.connections.length).to.equal(0);
			expect(n.connections.length).to.equal(0);
			expect(o.connections.length).to.equal(0);
			expect(m.influences.length).to.equal(0);
			expect(n.influences.length).to.equal(0);
			expect(o.influences.length).to.equal(0);
			o.connect(n);
			expect(n.influences.length).to.equal(1);
			expect(m.influences.length).to.equal(0);
			expect(o.connections.length).to.equal(1);
			o.connect(m);
			expect(n.influences.length).to.equal(1);
			expect(m.influences.length).to.equal(1);
			expect(o.connections.length).to.equal(2);
			o.disconnect(m);
			expect(n.influences.length).to.equal(1);
			expect(m.influences.length).to.equal(0);
			expect(o.connections.length).to.equal(1);
			expect(typeof n.init).to.equal('function');
			expect(typeof n.activation).to.equal('function');
			expect(typeof n.propogate).to.equal('function');
			expect(typeof n.dAwrt).to.equal('function');
			expect(typeof n.adjust).to.equal('function');
		}

	});

	it('Tests that each neuron gives a positive value after being given a positive value', function(){

		for(var x = 0 ; x < allNeuronKinds.length; x++){
			var m = neuron({typeOfNeuron: allNeuronKinds[x], randomness: 'one' });
			var n = neuron({typeOfNeuron: allNeuronKinds[x], randomness: 'one' });
			var o = neuron({typeOfNeuron: allNeuronKinds[x], randomness: 'one' });
			o.connect(n);
			o.connect(m);
			m.init();
			n.init();
			o.init();
			m.activation(1);
			n.activation(1);
			expect(o.activation() > 0).to.equal(true)
			o.propogate(0);
			expect(n.propogate() > 0).to.equal(true);
			expect(m.propogate() > 0).to.equal(true);
		}

	});

	//Added this after getting ready to scream.
	it('Calculates the derivative relative to some input correctly.', function(){

		for(var x = 0; x < allNeuronKinds.length; x++){
			var m = neuron({typeOfNeuron: 'input'});
			var n = neuron({typeOfNeuron: 'input'});
			var o = neuron({typeOfNeuron: allNeuronKinds[x], randomness: 'flatPositive', cost: 'squaredError' });
			var p = neuron({typeOfNeuron: allNeuronKinds[x], randomness: 'flatPositive', cost: 'squaredError' });

			o.connect(n);
			o.connect(m);
			p.connect(o);
			m.init();
			n.init();
			o.init();
			p.init();

			n.activation(1)
			m.activation(1)
			o.activation()
			p.activation()

			//Get what we expect the derivative to be.
			var expectedChangeOne = o.dAwrt(m);
			//Get old value
			var oldValue = o.activation();
			//Change input by 0.01
			m.activation(0.99);
			//Get the new value
			var newValue = o.activation();
			//This is the difference between the linear predicted change, and the actual change.
			var dif = (oldValue - newValue) - (expectedChangeOne * 0.01);
			//console.log("d", dif, (oldValue - newValue), expectedChangeOne * 0.01)
			expect(dif < 0.00002).to.be.true
		}

	});


	it('Adjusts itself after being in error, so that the error is less.', function(){

		for(var x = 0 ; x < allNeuronKinds.length; x++){
			//console.log(allNeuronKinds[x]);
			var m = neuron({typeOfNeuron: 'input'});
			var n = neuron({typeOfNeuron: 'input'});
			var o = neuron({typeOfNeuron: allNeuronKinds[x], randomness: 'flatPositive', cost: 'squaredError' });
			var p = neuron({typeOfNeuron: allNeuronKinds[x], randomness: 'flatPositive', cost: 'squaredError' });
			
			o.connect(n);
			o.connect(m);
			p.connect(o);
			m.init();
			n.init();
			o.init();
			p.init();

			m.activation(1);
			n.activation(1);
			o.activation(); 
			p.activation()
			p.propogate(0);
			o.propogate();
			var value = p.a;
			var costDerivO = o.dCwrtA;
			var costDerivP = p.dCwrtA;
			o.propogate(0);
			for(y = 0; y < 100; y++){
				m.activation(1);
				n.activation(1);
				o.activation();
				p.activation();
				p.propogate(0);
				o.propogate()
				p.adjust(0.2);
				o.adjust(0.2);
			}
			expect(costDerivO > o.dCwrtA).to.equal(true)
			expect(costDerivP > p.dCwrtA).to.equal(true)
			expect(value > p.a)
		}

	});

	it('Can handle the task of converting from one-spot to binary', function(){

		this.timeout(20000);

		var allNeuronKinds = ['linear','leakyrelu','relu','tanh', 'sigmoid'];
		var allEtaValues = [0.2,0.2,0.2,0.2,2];

		for(var x = 0 ; x < allNeuronKinds.length; x++){

			//console.log(allNeuronKinds[x])

			neurons = [[],[],[]];
			for(var a = 0; a < 8; a++){
				neurons[0].push(neuron({typeOfNeuron: 'input'}));
			}
			for(var a = 0; a < 16; a++){
				neurons[1].push(neuron({typeOfNeuron: allNeuronKinds[x], randomness: 'flatProportionateZero', cost: 'squaredError' }));
			}
			for(var a = 0; a < 3; a++){
				neurons[2].push(neuron({typeOfNeuron: allNeuronKinds[x], randomness: 'flatProportionateZero', cost: 'squaredError' }));
			}

			//Connect each neuron to all the prior neurons
			for(var a = 1; a < neurons.length; a++){
				for(var b = 0; b < neurons[a-1].length; b++){
					for(var c = 0; c < neurons[a].length; c++){
						neurons[a][c].connect(neurons[a-1][b]);
					}
				}
			}

			//Initialize the neurons
			for(var a = 0; a < neurons.length; a++){
				for(var b = 0; b < neurons[a].length; b++){
					neurons[a][b].init();
				}
			}

			//Train em all, say, 500 times
			for(var a = 0; a < 1500; a++){

				//Get input / output.
				num = Math.round(Math.random()*7);
				oneHot = [0,0,0,0,0,0,0,0];
				oneHot[num] = 1;
				oneHot.reverse();
				binary = num.toString(2).split('').map(function(n){return parseInt(n)});
				while(binary.length < 3){binary = [0].concat(binary)}
			

				for(var b = 0; b < neurons[0].length; b++){
					neurons[0][b].activation(oneHot[b]);
				}	
				for(var b = 0; b < neurons[1].length; b++){
					neurons[1][b].activation();
				}
				for(var b = 0; b < 3; b++){
					neurons[2][b].activation();
					neurons[2][b].propogate(binary[b]);
				}
				for(var b= 0; b < neurons[1].length; b++){
					neurons[1][b].propogate();
				}
				for(var c = 2; c > 0; c--){
					for (var b = 0; b < neurons[c].length; b++){
						neurons[c][b].adjust(allEtaValues[x]);
					}
				}

			}

			//Check
			for(var a = 0; a < 8; a++){

				num = a
				oneHot = [0,0,0,0,0,0,0,0];
				oneHot[num] = 1;
				oneHot.reverse();
				binary = num.toString(2).split('').map(function(n){return parseInt(n)});
				while(binary.length < 3){binary = [0].concat(binary)}

				for(var b = 0; b < 8; b++){
					neurons[0][b].activation(oneHot[b]);
				}
				for(var b = 1; b < 3; b++){
					for(var c = 0; c < neurons[b].length; c++){
						neurons[b][c].activation();
					}
				}
				result = neurons[2].map(function(n){return n.a});

				for(var c = 0; c < 3; c++){
					expect(Math.abs(binary[c]-result[c]) < 0.1).to.equal(true)
				}
				

			}
		}
	});

	var allNeuronKinds = ['linear','leakyrelu','relu','tanh', 'sigmoid'];
	

	it('Can handle the task of converting from one-spot to binary, using SGD', function(){

		this.timeout(20000);

		var allNeuronKinds = ['linear','leakyrelu','relu','tanh', 'sigmoid'];
		var allEtaValues = [0.2,0.3,0.3,0.5,10];

		for(var x = 0 ; x < 5; x++){

			//console.log(allNeuronKinds[x])

			neurons = [[],[],[]];
			for(var a = 0; a < 8; a++){
				neurons[0].push(neuron({typeOfNeuron: 'input'}));
			}
			for(var a = 0; a < 16; a++){
				neurons[1].push(neuron({typeOfNeuron: allNeuronKinds[x], randomness: 'flatProportionateZero', cost: 'squaredError' }));
			}
			for(var a = 0; a < 3; a++){
				neurons[2].push(neuron({typeOfNeuron: allNeuronKinds[x], randomness: 'flatProportionateZero', cost: 'squaredError' }));
			}

			//Connect each neuron to all the prior neurons
			for(var a = 1; a < neurons.length; a++){
				for(var b = 0; b < neurons[a-1].length; b++){
					for(var c = 0; c < neurons[a].length; c++){
						neurons[a][c].connect(neurons[a-1][b]);
					}
				}
			}

			//Initialize the neurons
			for(var a = 0; a < neurons.length; a++){
				for(var b = 0; b < neurons[a].length; b++){
					neurons[a][b].init();
				}
			}

			//Train em all, say, 500 times
			for(var a = 0; a < 2000; a++){

				//Get input / output.
				num = Math.round(Math.random()*7);
				oneHot = [0,0,0,0,0,0,0,0];
				oneHot[num] = 1;
				oneHot.reverse();
				binary = num.toString(2).split('').map(function(n){return parseInt(n)});
				while(binary.length < 3){binary = [0].concat(binary)}			

				for(var b = 0; b < neurons[0].length; b++){
					neurons[0][b].activation(oneHot[b]);
				}	
				for(var b = 0; b < neurons[1].length; b++){
					neurons[1][b].activation();
				}
				for(var b = 0; b < 3; b++){
					neurons[2][b].activation();
					neurons[2][b].propogate(binary[b]);
				}
				for(var b= 0; b < neurons[1].length; b++){
					neurons[1][b].propogate();
				}
				for(var c = 2; c > 0; c--){
					for (var b = 0; b < neurons[c].length; b++){
						neurons[c][b].getDeltas();
					}
				}

				if (a % 10 == 0){
					for(var c = 2; c > 0; c--){
						for (var b = 0; b < neurons[c].length; b++){
							//console.log(neurons[c][b].weightsDelta)
							neurons[c][b].applyDeltas(allEtaValues[x]);

						}
					}	
				}
			}

			//Check
			for(var a = 0; a < 8; a++){

				num = a
				oneHot = [0,0,0,0,0,0,0,0];
				oneHot[num] = 1;
				oneHot.reverse();
				binary = num.toString(2).split('').map(function(n){return parseInt(n)});
				while(binary.length < 3){binary = [0].concat(binary)}

				for(var b = 0; b < 8; b++){
					neurons[0][b].activation(oneHot[b]);
				}
				for(var b = 1; b < 3; b++){
					for(var c = 0; c < neurons[b].length; c++){
						neurons[b][c].activation();
					}
				}
				result = neurons[2].map(function(n){return n.a});

				for(var c = 0; c < 3; c++){
					expect(Math.abs(binary[c]-result[c]) < 0.1).to.equal(true)
				}
				

			}
		}
	});

});