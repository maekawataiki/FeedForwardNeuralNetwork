import java.io.*;

public class MNISTHandWritten {

	private int numLabels;
	private int numImages;
	private int numRows;
	private int numCols;
	private Tuple[] data;
	
	public static void main(String[] args) {
		MNISTHandWritten train = new MNISTHandWritten();
		train.MNISTReader("train_labels.idx1-ubyte", "train_images.idx3-ubyte", 40000);
		/*for(int x = 0; x < train.numRows; x++){
			for(int y = 0; y < train.numCols; y++){
				System.out.print(((((float[][])train.getData()[0].getA())[x*train.numRows+y][0]) == 0)? " " : "@");
			}
			System.out.println();
		}*/
		MNISTHandWritten test = new MNISTHandWritten();
		test.MNISTReader("t10k_labels.idx1-ubyte", "t10k_images.idx3-ubyte", 1000);
		int[] sizes = {784, 50, 50, 10};
		Network net = new Network(sizes);
		net.SGD(train.data, 30, 10, (float)5.0, test.data);
	}
	
	public MNISTHandWritten(){
	}

	public void MNISTReader(String labelFilename, String imageFilename, int size) {
		try {
			DataInputStream labels = new DataInputStream(new FileInputStream(labelFilename));
			DataInputStream images = new DataInputStream(new FileInputStream(imageFilename));
			// check data readable
			int magicNumber = labels.readInt();
			if (magicNumber != 2049) {
				System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
			}
			magicNumber = images.readInt();
			if (magicNumber != 2051) {
				System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
			}
			// check data is appropriate set
			this.numLabels = labels.readInt();
			this.numImages = images.readInt();
			this.numRows = images.readInt();
			this.numCols = images.readInt();
			if (numLabels != numImages) {
				StringBuilder str = new StringBuilder();
				str.append("Image file and label file do not contain the same number of entries.\n");
				str.append("  Label file contains: " + numLabels + "\n");
				str.append("  Image file contains: " + numImages + "\n");
				System.err.println(str.toString());
			}
			if (this.numLabels < size){
				System.err.println("do not have enough data");
			}
			// read data
			byte[] labelsData = new byte[numLabels];
			labels.read(labelsData);
			int imageVectorSize = numCols * numRows;
			byte[] imagesData = new byte[numLabels * imageVectorSize];
			images.read(imagesData);
			// translate
			this.data = new Tuple[size];
			int imageIndex = 0;
			for(int i = 0; i < size; i++) {
				int label = labelsData[i];
				float[][] inputData = new float[imageVectorSize][1];
				for(int j = 0; j < imageVectorSize; j++) {
					inputData[j][0] = (float) (imagesData[imageIndex++]&0xff);
				}
				this.data[i] = new Tuple(inputData, label);
			}
			images.close();
			labels.close();
		} catch (IOException ex) {
			System.err.println(ex);
		}
	}

	public int getNumLabels() {
		return numLabels;
	}

	public int getNumImages() {
		return numImages;
	}
	
	public int getNumRows() {
		return numRows;
	}

	public int getNumCols() {
		return numCols;
	}
	
	public Tuple[] getData(){
		return data;
	}
}
