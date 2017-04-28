import java.util.Random;
import java.util.function.DoubleBinaryOperator;
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
	float[][][] biases, weights;
	
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
		this.biases = new float[this.num_layers-1][][];
		for(int x = 0; x < this.num_layers - 1; x++){
			this.biases[x] = (gaussian_matrix(sizes[x+1], 1));
		}
		this.weights = new float[this.num_layers-1][][];
		for(int x = 0; x < this.num_layers-1; x++){
			this.weights[x] = (gaussian_matrix(sizes[x+1], sizes[x]));
		}
	}
	
	public float[][] feedforward(float[][] a){
		// Return the output of the network if "a" is input
		DoubleBinaryOperator d = (x, y) -> (x + y);
		for(int x = 0; x < biases.length; x++){
			a = sigmoid_vec(matrix_function(dot_product(weights[x], a), biases[x], d));
		}
		return a;
	}
	
	public void SGD(Tuple[] training_data, int epochs, int mini_batch_size, float eta, Tuple[] test_data){
		// Train the neural network using mini-batch stochastic
		// gradient descent. The "training_data" is a list of tuples
		// "(x, y)" representing the training inputs and the desired
		// outputs. The other non-optional parameters are 
		// self-explanatory. If "test_data" is provided then the 
		// network will be evaluated against the test data after each
		// epoch, and partial progress printed out. This is useful for
		// tracking progress, but slows things down substantially.
		int n_test = (test_data != null)? test_data.length : 0;
		int n = training_data.length;
		for(int x = 0; x < epochs; x++){
			shuffle(training_data);
			Tuple[] mini_batch = new Tuple[mini_batch_size];
			for(int y = 0; y < training_data.length / mini_batch_size; y++){
				for(int z = 0; z < mini_batch_size; z++){
					mini_batch[z] = training_data[y * mini_batch_size + z];
				}
				update_mini_batch(mini_batch, eta);
			}
			if(test_data!=null){
				System.out.println("Epoch " + x + ": " + evaluate(test_data) + " / " + n_test);
			}else{
				System.out.println("Epoch " + x + "complete");
			}
		}
	}
	
	public void update_mini_batch(Tuple[] mini_batch, float eta){
		// Update the network's weights and biases by applying 
		// gradient descent using back propagation to a single mini batch.
		// The "mini_batch" is a list of tuples"(x, y)", and "eta"
		// is the learning rate
		float[][][] nabla_b = zero_matrixes(biases);
		float[][][] nabla_w = zero_matrixes(weights);
		DoubleBinaryOperator d = (x, y) -> (x+y);
		for(int x = 0; x < mini_batch.length; x++){
			Tuple result = backprop((float[][])mini_batch[x].getA(), (int)mini_batch[x].getB());
			float[][][] delta_nabla_b = (float[][][])result.getA();
			float[][][] delta_nabla_w = (float[][][])result.getB();
			for(int y = 0; y < num_layers-1; y++){
				nabla_b[y] = matrix_function(nabla_b[y], delta_nabla_b[y], d);
				nabla_w[y] = matrix_function(nabla_w[y], delta_nabla_w[y], d);
			}
		}
		d = (x, y) -> (x-(eta/mini_batch.length)*y);
		for(int x = 0; x < num_layers-1; x++){
			weights[x] = matrix_function(weights[x], nabla_w[x], d);
			biases[x] = matrix_function(biases[x], nabla_b[x], d);
		}
	}
	
	public Tuple backprop(float[][] x, int y){
		// Return a tuple "(nabla_b, nabla_w)" representing the
		// gradient for the cost function C_x. "nabla_b" and 
		// "nabla_w" are layer-by-layer lists of arrays
		float[][][] nabla_b = zero_matrixes(biases);
		float[][][] nabla_w = zero_matrixes(weights);
		// feed forward
		float[][] activation = x;
		float[][][] activations = new float[this.num_layers][][]; // list to store all the activations, layer by layer
		activations[0] = x;
		float[][][] zs = new float[this.num_layers-1][][]; // list to store all the z vectors, layer by layer
		DoubleBinaryOperator d = (m, n) -> (m + n);
		for(int a = 0; a < num_layers-1; a++){
			float[][] z = matrix_function(dot_product(weights[a], activation), biases[a], d);
			zs[a] = z;
			activation = sigmoid_vec(z);
			activations[a+1] = activation;
		}
		// backward pass
		d = (a, b) -> (a * b);
		float[][] delta = matrix_function(cost_derivative(activations[activations.length-1], y), 
				sigmoid_prime_vec(zs[zs.length-1]), d);
		nabla_b[nabla_b.length-1] = delta;
		nabla_w[nabla_w.length-1] = dot_product(delta, transpose(activations[activations.length-2]));
		for(int l = 2; l < num_layers; l++){
			float[][] z = zs[zs.length-l];
			float[][] spv = sigmoid_prime_vec(z);
			delta = matrix_function(dot_product(transpose(weights[weights.length-l+1]), delta), spv, d);
			nabla_b[nabla_b.length-l] = delta;
			nabla_w[nabla_w.length-l] = dot_product(delta, transpose(activations[activations.length-l-1]));
		}
		Tuple result = new Tuple(nabla_b, nabla_w);;
		return result;
	}
	
	public int evaluate(Tuple[] test_data){
		// Return the number of test inputs for which the neural 
		// network outputs the correct result. Note that the neural
		// network's output is assumed to be the index of whichever
		// neuron in the final layer has the highest activation.
		int result = 0;
		for(int x = 0; x < test_data.length; x++){
			if(argmax(feedforward((float[][])test_data[x].getA())) == (int)test_data[x].getB())
				result++;
		}
		return result;
	}
	
	public float[][] cost_derivative(float[][] output_activations, int y){
		// Return the vector of partial derivatives \partial C_x /
		// \partial a for the output activations.
		float[][] result = new float[output_activations.length][1];
		result[y][0] = output_activations[y][0]-1;
		return result;
	}
	
	//****Miscellaneous functions****************************************************
	
	private float sigmoid(float z){
		//The sigmoid function
		return (float) (1.0/(1.0+Math.exp(-z)));
	}
	
	private float[][] sigmoid_vec(float[][] z){
		for(int x = 0; x < z.length; x++){
			for(int y = 0; y < z[0].length; y++){
				z[x][y] = sigmoid(z[x][y]);
			}
		}
		return z;
	}
	
	private float sigmoid_prime(float z){
		// Derivative of the sigmoid function
		return sigmoid(z)*(1-sigmoid(z));
	}
	
	private float[][] sigmoid_prime_vec(float[][] z){
		for(int x = 0; x < z.length; x++){
			for(int y = 0; y < z[0].length; y++){
				z[x][y] = sigmoid_prime(z[x][y]);
			}
		}
		return z;
	}
	
	private float[][] gaussian_matrix(int r, int c){
		float[][] m = new float[r][c];
		for(int x = 0; x < m.length; x++){
			for(int y = 0; y < m[0].length; y++){
				m[x][y] = (float)rnd.nextGaussian();
			}
		}
		return m;
	}
	
	private float[][] dot_product(float[][] x, float[][] y){
		if(x[0].length != y.length) System.err.println("Error: Dot Product");
		float[][] result = new float[x.length][y[0].length];
		for(int a = 0; a < x.length; a++){
			for(int b = 0; b < y[0].length; b++){
				for(int c = 0; c < y.length; c++){
					result[a][b] += x[a][c] * y[c][b];
				}
			}
		}
		return result;
	}
	
	private float[][] matrix_function(float[][] x, float[][] y, DoubleBinaryOperator d){
		if(x.length != y.length || x[0].length != y[0].length)
			System.err.println("Error: Matrix Function");
		float[][] sum = new float[x.length][x[0].length];
		for(int a = 0; a < sum.length; a++){
			for(int b = 0; b < sum[0].length; b++){
				sum[a][b] = (float) d.applyAsDouble(x[a][b], y[a][b]);
			}
		}
		return sum;
	}
	
	private float[][] transpose(float[][] m){
		float result[][] = new float[m[0].length][m.length];
		for(int x = 0; x < m.length; x++){
			for(int y = 0; y < m[0].length; y++){
				result[y][x] = m[x][y];
			}
		}
		return result;
	}
	
	private int argmax(float[][] x){
		int result = -1;
		float so_far = -1;
		for(int a = 0; a < x.length; a++){
			if(x[a][0] > so_far){
				result = a;
				so_far = x[a][0];
			}
		}
		return result;
	}
	
	private void shuffle (Tuple[] array){
		Random rnd = new Random();
		for(int i = 0; i < array.length; i++){
			int index = rnd.nextInt(array.length - i);
			Tuple a = array[index + i];
			array[index + i] = array[i];
			array[i] = a;
		}
	}
	
	private float[][][] zero_matrixes(float[][][] m){
		float[][][] result = new float[m.length][][];
		for(int x = 0; x < m.length; x++){
			result[x] = new float[m[x].length][m[x][0].length];
		}
		return result;
	}
}
