import java.util.LinkedList;
import java.util.Random;

/**
 * Neural Network Algorithm in Java
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
		// ever used in comuting the outputs from later layers.
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
		for(int x = 0; x < biases.size(); x++){
			a = sigmoid_vec(matrix_sum(dot_product(weights.get(x), a), biases.get(x)));
		}
		return a;
	}
	
	public void SGD(){
		// Train the neural network using mini-batch stochastic
		// gradient descent. The "training\data" is a list of tuples
		// "(x, y)" representing the training inputs and the desired
		// outputs. The other non-optional parameters are 
		// self-explanatory. If "test_data" is provided then the 
		// network will be evaluated against the test data after each
		// epoch, and partial progress printed out. This is useful for
		// tracking progress, but slows things down substantially.
	}
	
	public void update_mini_batch(){
		// Update the network's weights and biases by applying 
		// gradient descent using back propagation to a single mini batch.
		// The "mini_batch" is a list of tuples"(x, y)", and "eta"
		// is the learning rate
	}
	
	public void backprop(){
		// Return a tuple "(nabla_b, nabla_w)" representing the
		// gradient for the cost function C_x. "nabla_b" and 
		// "nabla_w" are layer-by-layer lists of arrays
	}
	
	public void evaluate(){
		// Return the number of test inputs for which the neural 
		// network outputs the correct result. Note that the neural
		// network's output is assumed to be the index of whichever
		// neuron in the final layer has the highest activation.
	}
	
	public void cost_derivative(){
		// Return the vector of partial derivatives \partial C_x /
		// \partial a for the output activations.
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
	
	private double[][] matrix_sum(double[][] x, double[][] y){
		if(x.length != y.length || x[0].length != 1 ||  y[0].length != 1)
			System.err.println("Error: Dot Sum");
		double[][] sum = new double[x.length][1];
		for(int a = 0; a < sum.length; a++){
			sum[a][0] = x[a][0] + y[a][0];
		}
		return sum;
	}
}
