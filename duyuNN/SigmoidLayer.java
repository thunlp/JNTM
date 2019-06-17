package duyuNN;

import java.util.Random;

public class SigmoidLayer implements NNInterface{

	public int length;
	public int linkId;
	public double[] input;
	public double[] inputG;
	public double[] output;
	public double[] outputG;
	
	public SigmoidLayer() {
	}
	
	public SigmoidLayer(int xLength)
	{
		this(xLength, 0);
	}
	
	public SigmoidLayer(int xLength, int xLinkId)
	{
		length = xLength;
		linkId = xLinkId;
		input = new double[length];
		inputG = new double[length];
		output = new double[length];
		outputG = new double[length];
	}
	
	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		//output[i] = 1 / (1 + Math.exp(-input[i]) );
		for (int i = 0; i < length; i++)
		{
	        if (input[i] > 0)
	        {
	            double x = (float)Math.exp(-1.0 * input[i]);
	
	            output[i] = 1.0f / (1.0f + x);
	        }
	        else
	        {
	            double x = (float)Math.exp(input[i]);
	
	            output[i] = x / (x + 1.0f);
	        }
		}
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		for (int i = 0; i < outputG.length; ++i)
        {
            inputG[i] = outputG[i] * output[i] * (1.0f - output[i]);
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
			inputG[i] = 0;
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
		return input;
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return inputG;
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		SigmoidLayer clone = new SigmoidLayer(length, linkId);
		return clone;
	}

}
