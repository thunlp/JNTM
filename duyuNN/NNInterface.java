package duyuNN;

import java.util.Random;

public interface NNInterface {

	public void randomize(Random r, double min, double max);
	
	public void forward();
	
	public void backward();
	
	public void update(double learningRate);
	
	public void updateAdaGrad(double learningRate, int batchsize);
	
	public void clearGrad();
	
	public void link(NNInterface nextLayer, int id) throws Exception;

    public void link(NNInterface nextLayer) throws Exception;

    public Object getInput(int id);

    public Object getOutput(int id);

    public Object getInputG(int id);

    public Object getOutputG(int id);
    
    public Object cloneWithTiedParams();
}
