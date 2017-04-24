import java.util.LinkedList;
import java.util.Map;
import java.util.Random;
import java.util.function.DoubleBinaryOperator;
import java.util.AbstractMap.SimpleEntry;
/**
 * General Neural Network Algorithm in Java
 * Consulted the python work by Michael Nielsen
 * Git: https://github.com/mnielsen/neural-networks-and-deep-learning
 * @author Maekawa
 * 
 */
public class Network {
	
	Random rnd = new Random();
	int num_layers;
	int[] sizes;
	LinkedList<double[][]> biases, weights;
	
	public Network(int[] sizes){
		// The list "sizes" contains the number of neurons in the 
		// respective layers of the network. For example, if the list
		// was [2,3,1] then it would be a three-layer network, with the
		// first layer containing 2 neurons, the second layer 3 neurons,
		// and the third layer 1 neuron. The biases and weights for the
		// network are initialized randomly using a Gaussian distribution
		// with mean 0, and variance 1. Note that the first 
		// layer is assumed to be an input layer, and by convention we
		// won't set any biases for those neurons, since biases are only
		// ever used in computing the outputs from later layers.
		this.num_layers = sizes.length;
		this.sizes = sizes;
		this.biases = new LinkedList<double[][]>();
		for(int x = 1; x < this.num_layers; x++){
			this.biases.add(gaussian_matrix(sizes[x], 1));
		}
		this.weights = new LinkedList<double[][]>();
		for(int x = 0; x < this.num_layers-1; x++){
			this.weights.add(gaussian_matrix(sizes[x+1], sizes[x]));
		}
	}
	
	public double[][] feedforward(double[][] a){
		// Return the output of the network if "a" is input
		DoubleBinaryOperator d = (x, y) -> x+y;
		for(int x = 0; x < num_layers-1; x++){
			a = sigmoid_vec(matrix_function(dot_product(weights.get(x), a), biases.get(x), d));
		}
		return a;
	}
	
	public void SGD(SimpleEntry<double[][], Integer>[] training_data, int epochs, 
			int mini_batch_size, double eta, SimpleEntry<double[][], Integer>[] test_data){
		// Train the neural network using mini-batch stochastic
		// gradient descent. The "training_data" is a list of tuples
		// "(x, y)" representing the training inputs and the desired
		// outputs. The other non-optional parameters are 
		// self-explanatory. If "test_data" is provided then the 
		// network will be evaluated against the test data after each
		// epoch, and partial progress printed out. This is useful for
		// tracking progress, but slows things down substantially.
		int n_test = (test_data!=null)? test_data.length : 0;
		int n = training_data.length;
		for(int x = 0; x < epochs / mini_batch_size; x++){
			shuffle(training_data);
			SimpleEntry<double[][], Integer>[] mini_batch = (SimpleEntry<double[][], Integer>[]) new Map[mini_batch_size];
			for(int y = 0; y < mini_batch_size; y++){
				mini_batch[y] = training_data[x * mini_batch_size + y];
			}
			update_mini_batch(mini_batch, eta);
			if(test_data!=null){
				System.out.println("Epoch " + x + ": " + evaluate(test_data) + " / " + n_test);
			}else{
				System.out.println("Epoch " + x + "complete");
			}
		}
	}
	
	public void update_mini_batch(SimpleEntry<double[][], Integer>[] mini_batch, double eta){
		// Update the network's weights and biases by applying 
		// gradient descent using back propagation to a single mini batch.
		// The "mini_batch" is a list of tuples"(x, y)", and "eta"
		// is the learning rate
		LinkedList<double[][]> nabla_b = zero_matrixes(biases);
		LinkedList<double[][]> nabla_w = zero_matrixes(weights);
		DoubleBinaryOperator d = (x, y) -> (x+y);
		for(int x = 0; x < mini_batch.length; x++){
			SimpleEntry<LinkedList<double[][]>, LinkedList<double[][]>> result = backprop(mini_batch[x].getKey(),mini_batch[x].getValue());
			LinkedList<double[][]> delta_nabla_b = result.getKey();
			LinkedList<double[][]> delta_nabla_w = result.getValue();
			for(int y = 0; y < num_layers-1; y++){
				nabla_b.add(y, matrix_function(nabla_b.get(y), delta_nabla_b.get(y), d));
				nabla_b.remove(y+1);
				nabla_w.add(y, matrix_function(nabla_w.get(y), delta_nabla_w.get(y), d));
				nabla_w.remove(y+1);
			}
		}
		d = (x, y) -> (x-(eta/mini_batch.length)*y);
		for(int x = 0; x < num_layers-1; x++){
			weights.add(x, matrix_function(weights.get(x), nabla_w.get(x), d));
			weights.remove(x+1);
			biases.add(x, matrix_function(biases.get(x), nabla_b.get(x), d));
			biases.remove(x+1);
		}
	}
	
	public SimpleEntry<LinkedList<double[][]>, LinkedList<double[][]>> backprop(double[][] x, int y){
		// Return a tuple "(nabla_b, nabla_w)" representing the
		// gradient for the cost function C_x. "nabla_b" and 
		// "nabla_w" are layer-by-layer lists of arrays
		LinkedList<double[][]> nabla_b = zero_matrixes(biases);
		LinkedList<double[][]> nabla_w = zero_matrixes(weights);
		// feed forward
		double[][] activation = x;
		LinkedList<double[][]> activations = new LinkedList<double[][]>(); // list to store all the activations, layer by layer
		activations.add(x);
		LinkedList<double[][]> zs = new LinkedList<double[][]>(); // list to store all the z vectors, layer by layer
		DoubleBinaryOperator d = (m, n) -> (m+n);
		for(int a = 0; a < num_layers-1; a++){
			double[][] z = matrix_function(dot_product(weights.get(a), activation), biases.get(a), d);
			zs.add(z);
			activation = sigmoid_vec(z);
			activations.add(activation);
		}
		// backward pass
		d = (a, b) -> (a*b);
		double[][] delta = matrix_function(cost_derivative(activations.get(num_layers-1), y), 
				sigmoid_prime_vec(zs.getLast()), d);
		nabla_b.removeLast();
		nabla_b.add(delta);
		nabla_w.removeLast();
		nabla_w.add(dot_product(delta, transpose(activations.get(num_layers-2))));
		for(int l = 2; l < num_layers; l++){
			double[][] z = zs.get(num_layers-l);
			double[][] spv = sigmoid_prime_vec(z);
			delta = matrix_function(dot_product(transpose(weights.get(num_layers-l+1)), delta), spv, d);
			nabla_b.add(num_layers-l, delta);
			nabla_b.remove(num_layers - l + 1);
			nabla_w.add(num_layers-l, dot_product(delta, transpose(activations.get(num_layers-l-1))));
			nabla_w.remove(num_layers - l + 1);
		}
		SimpleEntry<LinkedList<double[][]>, LinkedList<double[][]>> result = 
				new SimpleEntry<LinkedList<double[][]>, LinkedList<double[][]>>(nabla_b, nabla_w);
		return result;
	}
	
	public int evaluate(SimpleEntry<double[][], Integer>[] test_data){
		// Return the number of test inputs for which the neural 
		// network outputs the correct result. Note that the neural
		// network's output is assumed to be the index of whichever
		// neuron in the final layer has the highest activation.
		int result = 0;
		for(int x = 0; x < test_data.length; x++){
			if(argmax(feedforward(test_data[x].getKey())) == test_data[x].getValue())
				result++;
		}
		return result;
	}
	
	public double[][] cost_derivative(double[][] output_activations, int y){
		// Return the vector of partial derivatives \partial C_x /
		// \partial a for the output activations.
		double[][] result = new double[output_activations.length][1];
		result[y][0] = output_activations[y][0]-1;
		return result;
	}
	
	//****Miscellaneous functions****************************************************
	
	private double sigmoid(double z){
		//The sigmoid function
		return 1.0/(1.0+Math.exp(-z));
	}
	
	private double[][] sigmoid_vec(double[][] z){
		for(int x = 0; x < z.length; x++){
			for(int y = 0; y < z[0].length; y++){
				z[x][y] = sigmoid(z[x][y]);
			}
		}
		return z;
	}
	
	private double sigmoid_prime(double z){
		// Derivative of the sigmoid function
		return sigmoid(z)*(1-sigmoid(z));
	}
	
	private double[][] sigmoid_prime_vec(double[][] z){
		for(int x = 0; x < z.length; x++){
			for(int y = 0; y < z[0].length; y++){
				z[x][y] = sigmoid_prime(z[x][y]);
			}
		}
		return z;
	}
	
	private double[][] gaussian_matrix(int r, int c){
		double[][] m = new double[r][c];
		for(int x = 0; x < m.length; x++){
			for(int y = 0; y < m[0].length; y++){
				m[x][y] = rnd.nextGaussian();
			}
		}
		return m;
	}
	
	private double[][] dot_product(double[][] x, double[][] y){
		if(x[0].length != y.length) System.err.println("Error: Dot Product");
		double[][] result = new double[x.length][y[0].length];
		for(int a = 0; a < x.length; a++){
			for(int b = 0; b < y[0].length; b++){
				for(int c = 0; c < y.length; c++){
					result[a][b] += x[a][c] * y[c][b];
				}
			}
		}
		return result;
	}
	
	private double[][] matrix_function(double[][] x, double[][] y, DoubleBinaryOperator d){
		if(x.length != y.length || x[0].length != y[0].length)
			System.err.println("Error: Matrix Sum");
		double[][] sum = new double[x.length][x[0].length];
		for(int a = 0; a < sum.length; a++){
			for(int b = 0; b < sum[0].length; b++){
				sum[a][b] = d.applyAsDouble(x[a][b], y[a][b]);
			}
		}
		return sum;
	}
	
	private double[][] transpose(double[][] m){
		double result[][] = new double[m[0].length][m.length];
		for(int x = 0; x < m.length; x++){
			for(int y = 0; y < m.length; y++){
				result[y][x] = m[x][y];
			}
		}
		return result;
	}
	
	private int argmax(double[][] x){
		int result = -1;
		double so_far = -1;
		for(int a = 0; a < x.length; a++){
			if(x[a][0] > so_far){
				result = a;
				so_far = x[a][0];
			}
		}
		return result;
	}
	
	private void shuffle (SimpleEntry<double[][], Integer>[] array){
		// Shuffle by Fisher-Yates shuffle
		Random rnd = new Random();
		for(int i = array.length - 1; i > 0; i--){
			int index = rnd.nextInt(i+1);
			SimpleEntry<double[][], Integer> a = array[index];
			array[index] = array[i];
			array[i] = a;
		}
	}
	
	private LinkedList<double[][]> zero_matrixes(LinkedList<double[][]> m){
		LinkedList<double[][]> result = new LinkedList<double[][]>();
		for(int x = 0; x < num_layers-1; x++){
			result.add(new double[m.get(x).length][m.get(x)[0].length]);
		}
		return result;
	}
}
