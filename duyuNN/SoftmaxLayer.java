package duyuNN;

import java.util.Random;

public class SoftmaxLayer implements NNInterface{

	public int length;
	public int linkId;
	public double[] input;
	public double[] inputG;
	public double[] output;
	public double[] outputG;
	
	public SoftmaxLayer()
	{
	}

	public SoftmaxLayer(int xLength)
	{
		this(xLength, 0);
	}
	
	public SoftmaxLayer(int xLength, int xLinkId)
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
		double max = input[0];

        for (int i = 1; i < input.length; ++i)
        {
            if (input[i] > max)
            {
                max = input[i];
            }
        }

        double sum = 0;

        for (int i = 0; i < input.length; ++i)
        {
            output[i] = Math.exp(input[i] - max);
            sum += output[i];
        }

        for (int i = 0; i < input.length; ++i)
        {
            output[i] /= sum;
        }
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		
		for(int i = 0; i < length; i++)
		{
			inputG[i] = 0;
		}
		
		for(int i = 0; i < length; i++)
		{
			if (outputG[i] == 0)
            {
                continue;
            }
			
			for(int j = 0; j < length; j++)
			{
				if(i == j)
				{
					inputG[j] += outputG[i] * output[i] * (1 - output[j]);
				}
				else
				{
					inputG[j] += -outputG[i] * output[j] * output[i];
				}
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
		SoftmaxLayer clone = new SoftmaxLayer(length, linkId);
		return clone;
	}

}
