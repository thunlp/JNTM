package duyuNN;

import java.util.Random;

public class MultiConnectLayer implements NNInterface{

	public int[] inputLengths;

    public int outputLength;

    public double[][] input;

    public double[][] inputG;

    public double[] output;

    public double[] outputG;

    public int linkId;
	
    public MultiConnectLayer() {
	}
    
    public MultiConnectLayer(int[] xInputLengths)
    {
    	this(0, xInputLengths);
    }
    
    public MultiConnectLayer(int xLinkId, int[] xInputLengths)
    {
    	inputLengths = xInputLengths;
    	linkId = xLinkId;
    	
    	outputLength = 0;
    	for(int i = 0; i < inputLengths.length; i++)
    	{
    		outputLength += inputLengths[i];
    	}
    	
    	input = new double[inputLengths.length][];
    	inputG = new double[inputLengths.length][];
    	
    	for(int i = 0; i < inputLengths.length; i++)
    	{
    		input[i] = new double[inputLengths[i]];
    		inputG[i] = new double[inputLengths[i]];
    	}
    	
    	output = new double[outputLength];
    	outputG = new double[outputLength];
    }
    
	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		int k = 0;
		for(int i = 0; i < input.length; i++)
		{
			for(int j = 0; j < input[i].length; j++)
			{
				output[k] = input[i][j];
				k++;
			}
		}
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		int k = 0;
		for(int i = 0; i < input.length; i++)
		{
			for(int j = 0; j < input[i].length; j++)
			{
				inputG[i][j] = outputG[k];
				k++;
			}
		}
	}

	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		for(int i = 0; i < outputG.length; i++)
		{
			outputG[i] = 0;
		}
		
		for(int i = 0; i < inputG.length; i++)
		{
			for(int j = 0; j < inputG[i].length; j++)
			{
				inputG[i][j] = 0;
			}
		}
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != output.length || nextIG.length != outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		output = nextI;
		outputG = nextIG;
	}

	@Override
	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
		link(nextLayer, linkId);
	}

	@Override
	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return input[id];
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return inputG[id];
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		MultiConnectLayer clone = new MultiConnectLayer(linkId, inputLengths);
		return clone;
	}
}
