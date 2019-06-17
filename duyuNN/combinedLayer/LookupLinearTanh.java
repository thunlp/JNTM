package duyuNN.combinedLayer;

import java.util.Random;

import duyuNN.LinearLayer;
import duyuNN.LookupLayer;
import duyuNN.NNInterface;
import duyuNN.TanhLayer;

public class LookupLinearTanh implements NNInterface{
	
	public int windowSize;
	public int vocabSize;
	public int outputLength;
	public int embeddingLength;
	
	public LookupLayer lookup;
	public LinearLayer linear;
	public TanhLayer tanh;
	
	int linkId;
	
	public LookupLinearTanh()
	{
	}
	
	public LookupLinearTanh(LookupLayer seedLookup,
			LinearLayer seedLinear) throws Exception
	{
		vocabSize = seedLookup.vocabSize;
		outputLength = seedLinear.outputLength;
		embeddingLength = seedLookup.embeddingLength;
		windowSize = seedLookup.inputLength;
		
		lookup = (LookupLayer) seedLookup.cloneWithTiedParams();
		linear = (LinearLayer) seedLinear.cloneWithTiedParams();
		tanh = new TanhLayer(outputLength);
		
		lookup.link(linear);
		linear.link(tanh);
	}
	
	public LookupLinearTanh(
		int xWindowSize,
		int xVocabSize,
		int xOutputLength,
		int xEmbeddingLength) throws Exception
	{
		//vocabSize = xVocabSize;
		outputLength = xOutputLength;
		embeddingLength = xEmbeddingLength;
		windowSize = xWindowSize; // 1
		
		//lookup = new LookupLayer(embeddingLength, vocabSize, windowSize);
		linear = new LinearLayer(windowSize * embeddingLength, outputLength);
		tanh = new TanhLayer(outputLength);
		
		//lookup.link(linear);
		linear.link(tanh);
	}
	
	public void forward()
	{
		//lookup.forward();
		linear.forward();
		tanh.forward();
	}
	
	public void backward()
	{
		tanh.backward();
		linear.backward();
		//lookup.backward();
	}
	
	public void clearGrad()
	{
		//lookup.clearGrad();
		linear.clearGrad();
		tanh.clearGrad();
	}
	
	public LookupLinearTanh cloneWithTiedParams() 
	{
		LookupLinearTanh clone = new LookupLinearTanh();
		
		clone.vocabSize = vocabSize;
		clone.outputLength = outputLength;
		clone.embeddingLength = embeddingLength;
		clone.windowSize = windowSize;
		clone.linkId = linkId;
		
		clone.lookup = (LookupLayer)lookup.cloneWithTiedParams();
		clone.linear = (LinearLayer)linear.cloneWithTiedParams();
		clone.tanh = (TanhLayer)tanh.cloneWithTiedParams();
		
		try {
			clone.lookup.link(clone.linear);
			clone.linear.link(clone.tanh);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return clone;
	}
	
	public void link(NNInterface nextLayer, int id) throws Exception {
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != tanh.output.length || nextIG.length != tanh.outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		tanh.output = nextI;
		tanh.outputG = nextIG;
	}

	public void link(NNInterface nextLayer) throws Exception {
		link(nextLayer, linkId);
	}
	
	public Object getInput(int id) {
		//return lookup.input;
		return linear.input;
	}

	public Object getOutput(int id) {
		return tanh.output;
	}

	public Object getOutputG(int id) {
		return tanh.outputG;
	}

	@Override
	public void randomize(Random r, double min, double max) {
		linear.randomize(r, min, max);
	}
	
	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void update(double learningRate) {
		linear.update(learningRate);
	}

	public void regularizationLinear(double lambda) {
		linear.regularizationLinear(lambda);
	}
}
