package duyuNN;

import java.util.HashMap;
import java.util.Random;

public class LookupLayer implements NNInterface
{
	// y = Wx + b
	public double[][] table;
	
	public int embeddingLength;
	
	public int vocabSize;
	
	public int inputLength;
	
	public int[] input;
	
	public double[] output;
	
	public double[] outputG;
	
	public double[][] adaLR;
	
	public int linkId;
	
	HashMap<Integer, double[]> changeMap; 
	
	public LookupLayer()
	{
	}

	public LookupLayer(int xEmbeddingLength,
			int xVocabSize,
			int xInputLength)
	{
		this(xEmbeddingLength, xVocabSize, xInputLength, 0);
	}
	
	public LookupLayer(int xEmbeddingLength,
			int xVocabSize,
			int xInputLength,
			int xLinkId)
	{
		embeddingLength = xEmbeddingLength;
		vocabSize = xVocabSize;
		inputLength = xInputLength;
		
		/*table = new double[vocabSize][];
		for (int i = 0; i < table.length; ++i)
        {
            table[i] = new double[embeddingLength];
        }*/
		
		input   = new int[inputLength];
		output  = new double[embeddingLength * inputLength];
		outputG = new double[embeddingLength * inputLength];
		
		/*adaLR = new double[vocabSize][];
        for (int i = 0; i < adaLR.length; ++i)
        {
            adaLR[i] = new double[embeddingLength];
        }*/
        
        linkId = xLinkId;
	}
	
	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		for(int i = 0; i < vocabSize; i++)
		{
			for(int j = 0; j < embeddingLength; j++)
			{
				table[i][j] = r.nextFloat() * (max - min) + min;
			}
		}
	}

	@Override
	public void forward() 
	{
		// TODO Auto-generated method stub
		for (int i = 0; i < input.length; ++i)
        {
            int inputId = input[i];

            int offset = embeddingLength * i;

            if (inputId >= 0)
            {
                for (int j = 0; j < embeddingLength; ++j)
                {
                    output[offset + j] = table[inputId][j];
                }
            }
            else
            {
                for (int j = 0; j < embeddingLength; ++j)
                {
                    output[offset + j] = 0;
                }
            }
        }
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		if(null == changeMap)
		{
			changeMap = new HashMap<Integer, double[]>();
		}
		
		for (int i = 0; i < input.length; ++i)
        {
			int wordId = input[i];
			int offset = i * embeddingLength;
			
			if(changeMap.containsKey(wordId))
			{
				double[] embedG = changeMap.get(wordId);
				for (int j = 0; j < embeddingLength; ++j)
                {
                    embedG[j] += outputG[j + offset];
                }
			}
			else
			{
				double[] embedG = new double[embeddingLength];
				for (int j = 0; j < embeddingLength; ++j)
                {
                    embedG[j] = outputG[j + offset];
                }
				changeMap.put(wordId, embedG);
			}
        }
	}

	public void update(double learningRate, boolean isNorm) {
		// TODO Auto-generated method stub
		for(int wordId : changeMap.keySet())
		{
			double[] embedG = changeMap.get(wordId);
			double tmpL2Norm = 0.0;
			
			for(int j = 0; j < embeddingLength; j++)
			{
				table[wordId][j] += learningRate * embedG[j];
				
				if(isNorm)
				{
					tmpL2Norm = table[wordId][j] * table[wordId][j];
				}
			}
			
			if(tmpL2Norm > 1 && isNorm)
			{
				for(int j = 0; j < embeddingLength; j++)
				{
					table[wordId][j] = table[wordId][j] / tmpL2Norm;
				}
			}
		}
	}

	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		update(learningRate, false);
	}

	public void updateAdaGrad(double learningRate, 
			int batchsize, 
			boolean isNorm) {
		// TODO Auto-generated method stub
		for(int wordId : changeMap.keySet())
		{
			double[] embedG = changeMap.get(wordId);
			double tmpL2Norm = 0.0;
			
			for (int j = 0; j < embeddingLength; ++j)
            {
                adaLR[wordId][j] += 
                	(embedG[j] / batchsize) * (embedG[j] / batchsize);
            }
			
			for(int j = 0; j < embeddingLength; j++)
			{
				if(embedG[j] !=0)
					table[wordId][j] += learningRate * embedG[j] 
						/ (batchsize * Math.sqrt(adaLR[wordId][j]));
				
				if(isNorm)
				{
					tmpL2Norm = table[wordId][j] * table[wordId][j];
				}
			}
			
			if(tmpL2Norm > 1 && isNorm)
			{
				for(int j = 0; j < embeddingLength; j++)
				{
					table[wordId][j] = table[wordId][j] / tmpL2Norm;
				}
			}
		}
	}
	
	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		updateAdaGrad(learningRate, batchsize, false);
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		if (changeMap == null)
        {
            return;
        }
        changeMap.clear();
        
        for(int i = 0; i < outputG.length; i++)
        {
        	outputG[i] = 0;
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
		return null;
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		LookupLayer lookup = new LookupLayer();
		
		lookup.adaLR = adaLR;
		lookup.embeddingLength = embeddingLength;
		lookup.vocabSize = vocabSize;
		lookup.table = table;
		lookup.inputLength = inputLength;
		lookup.changeMap = new HashMap<Integer, double[]>();
		lookup.linkId = linkId;
		lookup.input = new int[inputLength];
		lookup.output = new double[output.length];
		lookup.outputG = new double[outputG.length];
		
		return lookup;
	}
	
	public void setEmbeddings(double[][] embed) throws Exception
    {
        if (embed == null || embed.length != table.length
            || embed.length == 0)
        {
            throw new Exception("embedding does not match!");
        }

        for (int i = 0; i < table.length; ++i)
        {
        	for(int j = 0; j < embeddingLength; j++)
        	{
        		table[i][j] = embed[i][j];
        	}
        }
    }

	public void regularizationLookup(double lambda) {
		for (int i = 0; i < table.length; ++i)
        {
        	for(int j = 0; j < embeddingLength; j++)
        	{
        		table[i][j] -= lambda * table[i][j];
        	}
        }
	}
}
