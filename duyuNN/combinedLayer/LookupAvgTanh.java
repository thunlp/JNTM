package duyuNN.combinedLayer;

import java.util.Random;

import duyuNN.AverageLayer;
import duyuNN.LinearLayer;
import duyuNN.LookupLayer;
import duyuNN.NNInterface;
import duyuNN.TanhLayer;

public class LookupAvgTanh implements NNInterface {

	public int windowSize;
	public int vocabSize;
	//public int outputLength;
	public int embeddingLength;
	
	public LookupLayer lookup;
	public AverageLayer average;
	public TanhLayer tanh;
	
	int linkId;
	
	public LookupAvgTanh(
			int xWindowSize, 	// # words in a sentence
			int xVocabSize,
	//		int xOutputLength,
			int xEmbeddingLength) throws Exception
		{
			vocabSize = xVocabSize;
	//		outputLength = xOutputLength;
			embeddingLength = xEmbeddingLength;
			windowSize = xWindowSize; 
			
			lookup = new LookupLayer(embeddingLength, vocabSize, windowSize);
			average = new AverageLayer(windowSize * embeddingLength, embeddingLength);
			tanh = new TanhLayer(embeddingLength);
			
			lookup.link(average);
			average.link(tanh);
		}
	
	public LookupAvgTanh() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		lookup.randomize(r, min, max);
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		lookup.forward();
		average.forward();
		tanh.forward();
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		tanh.backward();
		average.backward();
		lookup.backward();
	}

	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		lookup.update(learningRate, false); //false not norm
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		lookup.updateAdaGrad(learningRate, 1);
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		lookup.clearGrad();
		average.clearGrad();
		tanh.clearGrad();
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
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

	@Override
	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
		link(nextLayer, linkId);
	}

	@Override
	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return tanh.output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return tanh.outputG;
	}

	@Override
	public Object cloneWithTiedParams() { // only need the same table[][], i.e. location embeddings
		// TODO Auto-generated method stub
		
		/*LookupAvgTanh clone = new LookupAvgTanh();
		
		clone.vocabSize = vocabSize;
		//clone.outputLength = outputLength;
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
		
		return clone;*/
		return null;
	}

	public void regularizationLinear(double lambda) {
		lookup.regularizationLookup(lambda);
	}
}
