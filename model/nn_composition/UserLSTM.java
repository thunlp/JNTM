package model.nn_composition;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import duyuNN.MathOp;

public class UserLSTM extends Thread{
	static List<List<int[]>> trainData = new ArrayList<List<int[]>>();
	static List<List<int[]>> testData = new ArrayList<List<int[]>>();
	static HashSet<Integer> train_locs = new HashSet<Integer>(); //locations appeared in train set
	
	static public double[][] U; // each column embedding of locations
	static public double[][] W;
	static public double[][] V; //softmax
	static public double[][] V2;
	static public double[][] V3;
	static public double[][] NV;
	//static public double[][] R; //embedding of users
	
	static public double[][] UAdaLR, WAdaLR,VAdaLR,V2AdaLR,V3AdaLR,NVAdaLR;//,RAdaLR;
	
	static public double[] init_state;
	static public double[] dS;
	static public double[] SAdaLR;
	
	static public double[] init_C;
	static public double[] dC;
	static public double[] CAdaLR;
	
	static public double[][] dU,dV,dV2,dW,dV3,dNV;//,dR;
	
	static public double[][] R, RAdaLR, dR;
	static public double[][] NR, NRAdaLR, dNR;
	
	// LSTM parameters
	static public double[][] Wi1, Wi2, Wc1, Wc2, Wf1,Wf2;
	static public double[] bi,bc,bf;
	static public double[][] Wi1AdaLR, Wi2AdaLR, Wc1AdaLR, Wc2AdaLR, Wf1AdaLR,Wf2AdaLR;
	static public double[] biAdaLR,bcAdaLR,bfAdaLR;
	static public double[][] dWi1, dWi2, dWc1, dWc2, dWf1,dWf2;
	static public double[] dbi,dbc,dbf;
	static public double input_bias,cand_bias,forget_bias; // initialization of bi bc bf
	
	static public int hiddenLength;
	static public int loc_Num;
	static public int user_Num;
	static public int batchsize;
	
	static void loadData(String trainFile, String testFile) throws IOException
	{
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(trainFile)));
		String line;
		int last_uid = -1;
		List<int[]> ltmp = null;
		while((line = reader.readLine()) != null)
		{
			String[] stmp = line.split("\t");
			int uid = Integer.parseInt(stmp[0]);
			if(uid != last_uid)	//new user
			{
				if(last_uid != -1)
					trainData.add(ltmp);
				ltmp = new ArrayList<int[]>();
				last_uid = uid;
			}
			
			String[] stmp3 = stmp[1].split("\\s+");
			int[] data = new int[stmp3.length+1];
			data[0] = uid;

			for(int i=1;i<data.length;i++)
			{
				data[i] = Integer.parseInt(stmp3[i-1]);
				train_locs.add(data[i]);
			}
			ltmp.add(data);
		}
		trainData.add(ltmp);
		reader.close();
		
		last_uid = -1;
		reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(testFile)));
		while((line = reader.readLine()) != null)
		{
			String[] stmp = line.split("\t");
			int uid = Integer.parseInt(stmp[0]);
			if(uid != last_uid)	//new user
			{
				if(last_uid != -1)
					testData.add(ltmp);
				ltmp = new ArrayList<int[]>();
				last_uid = uid;
			}
			
			String[] stmp3 = stmp[1].split("\\s+");
			int[] data = new int[stmp3.length+1];
			data[0]=uid;
			for(int i=1;i<data.length;i++)
				data[i] = Integer.parseInt(stmp3[i-1]);
			ltmp.add(data);
		}
		testData.add(ltmp);
		reader.close();
	}
	
	static public double[][] newMatrix(int d1,int d2)
	{
		double[][] X = new double[d1][];
		for (int i = 0; i < d1; ++i)
			X[i] = new double[d2];
		return X;
	}
	
	static public void initialize(int embeddingLength,int locNum,int userNum)
	{
		hiddenLength = embeddingLength;
		loc_Num = locNum;
		user_Num = userNum;
		
		
        U = newMatrix(locNum,embeddingLength);
        UAdaLR = newMatrix(locNum,embeddingLength);
        dU = newMatrix(locNum,embeddingLength);
        
        W=newMatrix(embeddingLength,embeddingLength);
        WAdaLR=newMatrix(embeddingLength,embeddingLength);
        dW=newMatrix(embeddingLength,embeddingLength);
        
        V=newMatrix(locNum,embeddingLength);
        VAdaLR = newMatrix(locNum,embeddingLength);
        dV=newMatrix(locNum,embeddingLength);
        
        init_state = new double[embeddingLength];
        dS = new double[embeddingLength];
        SAdaLR = new double[embeddingLength];	
		
		
		V2=newMatrix(locNum,embeddingLength);
		V2AdaLR=newMatrix(locNum,embeddingLength);
		dV2=newMatrix(locNum,embeddingLength);
		
		R=newMatrix(userNum,embeddingLength);
		RAdaLR=newMatrix(userNum,embeddingLength);
		dR=newMatrix(userNum,embeddingLength);
		
		V3=newMatrix(locNum,embeddingLength);
        V3AdaLR=newMatrix(locNum,embeddingLength);
        dV3=newMatrix(locNum,embeddingLength);
        //lstm input gate
        Wi1 = newMatrix(embeddingLength,embeddingLength);
        Wi1AdaLR = newMatrix(embeddingLength,embeddingLength);
        dWi1 = newMatrix(embeddingLength,embeddingLength);
        Wi2 = newMatrix(embeddingLength,embeddingLength);
        Wi2AdaLR = newMatrix(embeddingLength,embeddingLength);
        dWi2 = newMatrix(embeddingLength,embeddingLength);
        bi = new double[embeddingLength];
        for(int i=0;i<embeddingLength;i++)	//alert
            bi[i] = input_bias;
        biAdaLR = new double[embeddingLength];
        dbi = new double[embeddingLength];
        //candidate
        Wc1 = newMatrix(embeddingLength,embeddingLength);
        Wc1AdaLR = newMatrix(embeddingLength,embeddingLength);
        dWc1 = newMatrix(embeddingLength,embeddingLength);
        Wc2 = newMatrix(embeddingLength,embeddingLength);
        Wc2AdaLR = newMatrix(embeddingLength,embeddingLength);
        dWc2 = newMatrix(embeddingLength,embeddingLength);
        bc = new double[embeddingLength];
        for(int i=0;i<embeddingLength;i++)	//alert
            bc[i] = cand_bias;
        bcAdaLR = new double[embeddingLength];
        dbc = new double[embeddingLength];
        // forget gate
        Wf1 = newMatrix(embeddingLength,embeddingLength);
        Wf1AdaLR = newMatrix(embeddingLength,embeddingLength);
        dWf1 = newMatrix(embeddingLength,embeddingLength);
        Wf2 = newMatrix(embeddingLength,embeddingLength);
        Wf2AdaLR = newMatrix(embeddingLength,embeddingLength);
        dWf2 = newMatrix(embeddingLength,embeddingLength);
        bf = new double[embeddingLength];
        for(int i=0;i<embeddingLength;i++)
            bf[i] = forget_bias;
        bfAdaLR = new double[embeddingLength];
        dbf = new double[embeddingLength];
        
        init_C = new double[embeddingLength];
        dC = new double[embeddingLength];
        CAdaLR = new double[embeddingLength];
		
        NR=newMatrix(userNum,embeddingLength);
        NRAdaLR=newMatrix(userNum,embeddingLength);
        dNR=newMatrix(userNum,embeddingLength);
        
        NV=newMatrix(locNum,embeddingLength);
        NVAdaLR=newMatrix(locNum,embeddingLength);
        dNV=newMatrix(locNum,embeddingLength);

	}
	
	static public void randomMatrix(double[][] X, Random r, double min, double max)
	{
		for(int i = 0; i < X.length; i++)
			for(int j = 0; j < X[i].length; j++)
				X[i][j] = r.nextFloat() * (max - min) + min;
	}
	
	static public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		
		randomMatrix(U,r,min,max);
		randomMatrix(W,r,min,max);
		randomMatrix(V,r,min,max);
		for(int i=0;i<init_state.length;i++)
			init_state[i] = r.nextFloat() * (max - min) + min;
		
		randomMatrix(V2,r,min,max);
		randomMatrix(R,r,min,max);
		
        randomMatrix(V3,r,min,max);
        randomMatrix(Wi1,r,min,max);
        randomMatrix(Wi2,r,min,max);
        randomMatrix(Wc1,r,min,max);
        randomMatrix(Wc2,r,min,max);
        randomMatrix(Wf1,r,min,max);
        randomMatrix(Wf2,r,min,max);		
        for(int i=0;i<init_C.length;i++)
            init_C[i] = r.nextFloat() * (max - min) + min;
    
        randomMatrix(NR,r,min,max);
        randomMatrix(NV,r,min,max);
		
	}
	

	static double probThreshold;
	static double learningRate;
	

	static double loss;
	static int num;//num of locations in data
	
	static private Lock lock1 = new ReentrantLock();
	static private Lock lock2 = new ReentrantLock();
	//static private Lock lock3 = new ReentrantLock();
	static int current = 0;
	//static int currentBatch = 0;
	//static int countData = 0;
	static public synchronized int getJob() {
		++current;
        return current-1;
    }
	
	static int current_id = -1;
	
	public void run()
	{
		int idxData;
		while((idxData=getJob())<trainData.get(current_id).size())
		{
			//System.out.println(idxData);
			if(idxData >= trainData.get(current_id).size())
				break;
			int[] data = trainData.get(current_id).get(idxData);
			
			handle(data,idxData);
			//System.out.println(idxData+"finish");
			//handle_fix_state(data);
	
		}
		this.stop();
	}
	
	public static double[][] dhiddenState; //grad ht
	
	public void handle(int[] data, int idxData) //forward and backward a rnn
	{
		//System.out.println(idxData);
		double[] state;
		List<double[]> allStates= new ArrayList<double[]>();
		List<double[]> allStateGs= new ArrayList<double[]>();
		int uid = data[0];
		int[] locs = new int[data.length-1];
		for(int i=0;i<locs.length;i++)
			locs[i] = data[i+1];	
		double tmploss = 0;
		int tmpnum=locs.length;
		
		//get user representation
		double[] userRepre = null;
				
        if(idxData==0)
            userRepre = tanh(init_C);
        else
            userRepre = hiddenState.get(idxData-1).get(hiddenState.get(idxData-1).size()-1);
		
		state = new double[hiddenLength];

        for(int i=0;i<hiddenLength;i++)
            state[i] = init_state[i];
        allStates.add(state);


		for(int i=0;i<locs.length;i++)
		{
			//forward to hidden state
			double[] next_state = new double[hiddenLength];
			
            if(i==0)
                MathOp.Axpy(W, state, next_state);
            else
                MathOp.Axpb(W, state, U[locs[i-1]], next_state);
            //tanh
            next_state = tanh(next_state);			
            state = next_state;
            allStates.add(state);
			

			//sample negative samples
			int positive = locs[i];
			List<Integer> neg_samples = new ArrayList<Integer>();
			Random rnd = new Random();
			for(int j=0;j<loc_Num;j++)
			{
				if(positive!=j && rnd.nextDouble()<0.01) //alert negative sample ratio is 0.01
					neg_samples.add(j);
			}
			// forward to output
			double[] output = new double[neg_samples.size()+1]; //all neg and 1 pos
			for(int j=0;j<neg_samples.size();j++)
			{
				output[j] = MathOp.dotProduct(V2[neg_samples.get(j)],R[uid]);

                output[j]+=MathOp.dotProduct(V[neg_samples.get(j)], state);

                output[j]+=MathOp.dotProduct(V3[neg_samples.get(j)],userRepre);

                output[j]+=MathOp.dotProduct(NV[neg_samples.get(j)],NR[uid]);
			}
			output[neg_samples.size()] = MathOp.dotProduct(V2[positive],R[uid]);

            output[neg_samples.size()]+=MathOp.dotProduct(V[positive], state);

            output[neg_samples.size()]+=MathOp.dotProduct(V3[positive],userRepre);

            output[neg_samples.size()]+=MathOp.dotProduct(NV[positive],NR[uid]);
			//softmax
			double max = output[0];
	        for (int j = 1; j < output.length; ++j)
	            if (output[j] > max)
	                max = output[j];

	        double sum = 0;
	        for (int j = 0; j < output.length; ++j)
	        {
	            output[j] = Math.exp(output[j] - max);
	            sum += output[j];
	        }
	        for (int j = 0; j < output.length; ++j)
	            output[j] /= sum;
	        //compute loss and grad
	        tmploss += -Math.log(output[neg_samples.size()]);
	        	//backward of softmax
	        double[] outputG = new double[output.length];//grad before softmax
	        double Gvalue = 0;//the only term which is not zero in grad after softmax
	        if(output[neg_samples.size()] < probThreshold)
	        	Gvalue =  1.0 / probThreshold;
			else
				Gvalue = 1.0 / output[neg_samples.size()];

	        for(int j = 0; j < output.length; j++)
				if(j == neg_samples.size()) //positive example
					outputG[j] += Gvalue * output[neg_samples.size()] * (1 - output[j]);
				else
					outputG[j] += -Gvalue * output[j] * output[neg_samples.size()];
	        
	        double[] stateG = new double[hiddenLength];
	        
            for(int k=0;k<hiddenLength;k++)
                for(int j=0;j<neg_samples.size();j++)
                {
                    stateG[k] += outputG[j] * V[neg_samples.get(j)][k];
                }
            for(int k=0;k<hiddenLength;k++)
            {
                stateG[k] += outputG[neg_samples.size()] * V[positive][k];
            }
            allStateGs.add(stateG);
	        
	        lock1.lock();
	        loss += tmploss;
	        num += tmpnum;
	        
	        
	        	//compute grad for V and state
	        for(int j=0;j<neg_samples.size();j++)
	        	for(int k=0;k<hiddenLength;k++)
	        	{

	        		dV[neg_samples.get(j)][k] += outputG[j]*state[k];
	        		dV2[neg_samples.get(j)][k] += outputG[j]*R[uid][k];
	        		dV3[neg_samples.get(j)][k] += outputG[j]*userRepre[k];
	        		dNV[neg_samples.get(j)][k] += outputG[j]*NR[uid][k];
	        	}
	        for(int k=0;k<hiddenLength;k++)
	        {
	        	dV[positive][k] += outputG[neg_samples.size()] * state[k];
	        	dV2[positive][k] += outputG[neg_samples.size()] * R[uid][k];
	        	dV3[positive][k] += outputG[neg_samples.size()] * userRepre[k];
	        	dNV[positive][k] += outputG[neg_samples.size()] * NR[uid][k];
	        }
	        lock1.unlock();	
	        
	      //changed
	        lock2.lock();
	        for(int k=0;k<hiddenLength;k++)
	        	for(int j=0;j<neg_samples.size();j++)
	        		dR[uid][k] += outputG[j] * V2[neg_samples.get(j)][k];

	        for(int k=0;k<hiddenLength;k++)
	        	dR[uid][k] += outputG[neg_samples.size()] * V2[positive][k];
	        lock2.unlock();
	        
	        double[] dtmp = new double[hiddenLength];
	        
            for(int k=0;k<hiddenLength;k++)
                for(int j=0;j<neg_samples.size();j++)
                {
                    dhiddenState[idxData][k] += outputG[j] * V3[neg_samples.get(j)][k];
                    //dtmp[k] += outputG[j] * V2[neg_samples.get(j)][k];
                }
            for(int k=0;k<hiddenLength;k++)
            {
                dhiddenState[idxData][k] += outputG[neg_samples.size()] * V3[positive][k];
                //dtmp[k] += outputG[neg_samples.size()] * V2[positive][k];
            }
        
            for(int k=0;k<hiddenLength;k++)
                for(int j=0;j<neg_samples.size();j++)
                    dNR[uid][k] += outputG[j] * NV[neg_samples.get(j)][k];

            for(int k=0;k<hiddenLength;k++)
                dNR[uid][k] += outputG[neg_samples.size()] * NV[positive][k];
	        
	        //added
	        //double[] drtmp = tanh_backward(dtmp,userRepre);
	       // dR[uid]=MathOp.sum(dR[uid], drtmp);
		}
	        
		
	    //recursively compute grad for U,W,init_state

		double[] last_stateG = new double[hiddenLength];
	    for(int j=locs.length-1;j>=0;j--)
        {
	    	double[] stateG = allStateGs.get(j);
        	double norm = 0;
        	for(int k=0;k<hiddenLength;k++)
        		norm+=stateG[k]*stateG[k];
        	norm = Math.sqrt(norm);
        	if(norm>15)
        	{
        		for(int k=0;k<hiddenLength;k++)
        			stateG[k] /= norm/15;
        	}
	    	stateG = MathOp.sum(stateG, last_stateG);
        	//backward from tanh
	    	stateG= tanh_backward(stateG,allStates.get(j+1));
        	
        	
        	double[] gtmp = new double[hiddenLength];
        	for(int k=0;k<hiddenLength;k++)
        	{
        		gtmp[k] = stateG[k];
        		stateG[k] = 0;
        	}
        	for(int k=0;k<hiddenLength;k++)	//update stateG
        	{
        		for(int iter=0;iter<hiddenLength;iter++)
        			stateG[k] += gtmp[iter] * W[iter][k];
        		last_stateG[k] = stateG[k];
        	}
        	lock2.lock();
        	//compute grad for U,W,R and state
        	if(j!=0)	//update dU
        	{
        		for (int k = 0; k < hiddenLength; ++k)
        			dU[locs[j-1]][k] += gtmp[k];
        	}
        	for(int iter=0;iter<hiddenLength;iter++) //update dW
	        	for(int k=0;k<hiddenLength;k++)
	        		dW[iter][k] += gtmp[iter]*allStates.get(j)[k];
        	
        	
        	if(j==0)	
        	{
        		for(int k=0;k<hiddenLength;k++)
        		{
        			dS[k] += stateG[k];
        		}
        	}
        	lock2.unlock();
        }
		
	}
	
	
	static public List<List<double[]>> userState = new ArrayList<List<double[]>>(); //ct
	//inter variable all input forget candidate values
	static public List<List<double[]>> inputState = new ArrayList<List<double[]>>();	//it
	static public List<List<double[]>> candState = new ArrayList<List<double[]>>();	//ct0
	static public List<List<double[]>> forgetState = new ArrayList<List<double[]>>(); //ft
	static public List<List<double[]>> hiddenState = new ArrayList<List<double[]>>(); //ht
	
	static public void lstm_forward()
	{
		dhiddenState = new double[trainData.get(current_id).size()][];
		for(int i=0;i<trainData.get(current_id).size();i++)
			dhiddenState[i] = new double[hiddenLength];
		
		int uid = trainData.get(current_id).get(0)[0];
		double[] last_ct = new double[hiddenLength];
		for(int i=0;i<hiddenLength;i++)
			last_ct[i] = init_C[i];	//C -1
		
		for(int i=0;i<trainData.get(current_id).size();i++)
		{
			List<double[]> subuserState = new ArrayList<double[]>();	//ct
			List<double[]> subinputState = new ArrayList<double[]>();
			List<double[]> subcandState = new ArrayList<double[]>();	//ct0
			List<double[]> subforgetState = new ArrayList<double[]>();
			List<double[]> subhiddenState = new ArrayList<double[]>();
			
			for(int j=1;j<trainData.get(current_id).get(i).length;j++) //skip uid
			{
				int lid = trainData.get(current_id).get(i)[j]; //location id
				//System.out.println(lid);
				double[] last_ht = tanh(last_ct);
				
				//input gate
				double[] it = new double[hiddenLength];
				MathOp.Axpb(Wi1, U[lid], bi, it);
				MathOp.Axpy(Wi2, last_ht, it);
				it = sigmoid(it);
				subinputState.add(it);
				
				//candidate
				double[] ct0 = new double[hiddenLength];
				MathOp.Axpb(Wc1, U[lid], bc, ct0);
				MathOp.Axpy(Wc2, last_ht, ct0);
				ct0 = tanh(ct0);				
				subcandState.add(ct0);
				
				//forget gate
				double[] ft = new double[hiddenLength];
				MathOp.Axpb(Wf1, U[lid], bf, ft);
				MathOp.Axpy(Wf2, last_ht, ft);
				ft = sigmoid(ft);
				subforgetState.add(ft);
				
				//new ct
				double[] ct = new double[hiddenLength];
				MathOp.pointwiseSum(it, ct0, ft, last_ct, ct);
				subuserState.add(ct);
				
				last_ct = ct;
				
				//ht
				subhiddenState.add(tanh(ct));
			}
			
			inputState.add(subinputState);
			candState.add(subcandState);
			forgetState.add(subforgetState);
			userState.add(subuserState);
			hiddenState.add(subhiddenState);
		}
	}
	
	public static void lstm_backward()
	{
		//dhiddenState bkward
		
		double[] lastht_grad = new double[hiddenLength];	// grad of h_{t-1}
		double[] lastct_grad = new double[hiddenLength];	// grad of c_{t-1}
		int uid = trainData.get(current_id).get(0)[0];
		//System.out.println(uid);
		
		for(int i=trainData.get(current_id).size()-1;i>0;i--)
		{
			//System.out.println(current_id);
			//System.out.println(i);
			lastht_grad = MathOp.sum(dhiddenState[i],lastht_grad);
			
			
			for(int j=hiddenState.get(i-1).size()-1;j>=0;j--)
			{
				//System.out.println("j"+j);
				double[] ht = hiddenState.get(i-1).get(j);			
				double[] dct = MathOp.sum(tanh_backward(lastht_grad,ht), lastct_grad);				
				
				double[] dit = MathOp.product(dct, candState.get(i-1).get(j));
				double[] dct0 = MathOp.product(dct,inputState.get(i-1).get(j));
				double[] lastct;	//ct-1
				if(j>0)
					lastct = candState.get(i-1).get(j-1);
				else if(j==0 && i>1)
					lastct = candState.get(i-2).get(candState.get(i-2).size()-1);
				else //j==0 i==1
					lastct = init_C;
				
				double[] dft = MathOp.product(dct, lastct);
				lastct_grad = MathOp.product(dct, forgetState.get(i-1).get(j));
				
				// bp it ft ct0				
				lastht_grad = new double[hiddenLength];
				double[] lastht = tanh(lastct);
				double[] dtmp = new double[hiddenLength];
				//System.out.println(j);
				//System.out.println(trainData.get(current_id).get(i-1).length);
				int lid = trainData.get(current_id).get(i-1)[j+1]; //+1 because uid
					//it
				double[] ditbk = sigmoid_backward(dit,inputState.get(i-1).get(j));
				for(int k=0;k<hiddenLength;k++)
					dbi[k] += ditbk[k];				
				MathOp.xdotA(ditbk, Wi1, dtmp);
				for(int k=0;k<hiddenLength;k++)
					dU[lid][k] += dtmp[k];
				MathOp.addto(dWi1, MathOp.outProduct(ditbk, U[lid]));
				MathOp.xdotA(ditbk, Wi2, dtmp);
				for(int k=0;k<hiddenLength;k++)
					lastht_grad[k] += dtmp[k];
				MathOp.addto(dWi2, MathOp.outProduct(ditbk, lastht));
					//ft
				double[] dftbk = sigmoid_backward(dft,forgetState.get(i-1).get(j));
				for(int k=0;k<hiddenLength;k++)
					dbf[k] += dftbk[k];				
				MathOp.xdotA(dftbk, Wf1, dtmp);
				for(int k=0;k<hiddenLength;k++)
					dU[lid][k] += dtmp[k];
				MathOp.addto(dWf1, MathOp.outProduct(dftbk, U[lid]));
				MathOp.xdotA(dftbk, Wf2, dtmp);
				for(int k=0;k<hiddenLength;k++)
					lastht_grad[k] += dtmp[k];
				MathOp.addto(dWf2, MathOp.outProduct(dftbk, lastht));
					//ct0
				double[] dct0bk = tanh_backward(dct0,candState.get(i-1).get(j));
				for(int k=0;k<hiddenLength;k++)
					dbc[k] += dct0bk[k];				
				MathOp.xdotA(dct0bk, Wc1, dtmp);
				for(int k=0;k<hiddenLength;k++)
					dU[lid][k] += dtmp[k];
				MathOp.addto(dWc1, MathOp.outProduct(dct0bk, U[lid]));
				MathOp.xdotA(dct0bk, Wc2, dtmp);
				for(int k=0;k<hiddenLength;k++)
					lastht_grad[k] += dtmp[k];
				MathOp.addto(dWc2, MathOp.outProduct(dct0bk, lastht));
			}
			
			//System.out.println("ie"+i);
			
		}

		lastht_grad = MathOp.sum(dhiddenState[0],lastht_grad);
		double[] dct = MathOp.sum(tanh_backward(lastht_grad,tanh(init_C)), lastct_grad);
		for(int k=0;k<hiddenLength;k++)
			dC[k] += dct[k];
		//	bk tanh
		//	only update R with dhidden and lastht ct grad;
	}
	
	public static void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub

		updateMatrix(U,UAdaLR,dU,learningRate,batchsize);
		updateMatrix(W,WAdaLR,dW,learningRate,batchsize);
		updateMatrix(V,VAdaLR,dV,learningRate,batchsize);
		updateVector(init_state,SAdaLR,dS,learningRate,batchsize);

		updateMatrix(V2,V2AdaLR,dV2,learningRate,batchsize);
		updateMatrix(R,RAdaLR,dR,learningRate,batchsize);
		
		
        updateMatrix(V3,V3AdaLR,dV3,learningRate,batchsize);
        updateMatrix(Wi1,Wi1AdaLR,dWi1,learningRate,batchsize);
        updateMatrix(Wi2,Wi2AdaLR,dWi2,learningRate,batchsize);	
        updateMatrix(Wc1,Wc1AdaLR,dWc1,learningRate,batchsize);
        updateMatrix(Wc2,Wc2AdaLR,dWc2,learningRate,batchsize);	
        updateMatrix(Wf1,Wf1AdaLR,dWf1,learningRate,batchsize);
        updateMatrix(Wf2,Wf2AdaLR,dWf2,learningRate,batchsize);	
        updateVector(bi,biAdaLR,dbi,learningRate,batchsize);
        updateVector(bc,biAdaLR,dbi,learningRate,batchsize);
        updateVector(bf,biAdaLR,dbi,learningRate,batchsize);
        updateVector(init_C,CAdaLR,dC,learningRate,batchsize);
    
        updateMatrix(NR,NRAdaLR,dNR,learningRate,batchsize);
        updateMatrix(NV,NVAdaLR,dNV,learningRate,batchsize);

		
	}
	
	public static void clearGrad()
	{

        clearMatrix(dU);
        clearMatrix(dW);
        clearMatrix(dV);
        clearVector(dS);

		clearMatrix(dV2);
		clearMatrix(dR);
		
		
        clearMatrix(dV3);
        clearMatrix(dWi1);
        clearMatrix(dWi2);
        clearMatrix(dWc1);
        clearMatrix(dWc2);
        clearMatrix(dWf1);
        clearMatrix(dWf2);
        clearVector(dbi);
        clearVector(dbc);
        clearVector(dbf);
        clearVector(dC);		
        clearMatrix(dNR);
        clearMatrix(dNV);

	}
	
	public static List<int[]> rank_prediction(List<int[]> userdata)
	{
		List<int[]> ranks = new ArrayList<int[]>();
		int uid = userdata.get(0)[0];
		int idx = -1;
		// get index in trainData
		for(int i=0;i<trainData.size();i++)
			if(trainData.get(i).get(0)[0] == uid)
			{
				idx=i;
				break;
			}
		
		Set<Integer> existLocs = new HashSet<Integer>();
		for(int i=0;i<trainData.get(idx).size();i++)
		{
			for(int j=1;j<trainData.get(idx).get(i).length;j++) //skip uid
				existLocs.add(trainData.get(idx).get(i)[j]);
		}
		
		current_id = idx;

		lstm_forward();
		
		double[] last_ct = null;
		double[] last_ht = null;
		double[] userRepre = null;

        last_ct = userState.get(userState.size()-1).get(userState.get(userState.size()-1).size()-1);
        last_ht = tanh(last_ct);
        userRepre = last_ht;

		for(int index=0;index<userdata.size();index++)
		{
			int[] data = userdata.get(index);
			int[] locs = new int[data.length-1];
			for(int i=0;i<locs.length;i++)
				locs[i] = data[i+1];		
		
			num=locs.length;
			int[] rank = new int[num*2];	//half for general prediction;half for new POI prediction 0,2,4...;1,3,5... -1 if not in train set ; -1 if not new
			
			double[] state = new double[hiddenLength];

            for(int i=0;i<hiddenLength;i++)
                state[i] = init_state[i];

			int last_pos = -1;
			
			for(int i=0;i<locs.length;i++)
			{
				//forward the user's lstm first
				int lid = locs[i]; //location id
								
                //input gate
                double[] it = new double[hiddenLength];
                MathOp.Axpb(Wi1, U[lid], bi, it);
                MathOp.Axpy(Wi2, last_ht, it);
                it = sigmoid(it);
                
                    //candidate
                double[] ct0 = new double[hiddenLength];
                MathOp.Axpb(Wc1, U[lid], bc, ct0);
                MathOp.Axpy(Wc2, last_ht, ct0);
                ct0 = tanh(ct0);				
                
                    //forget gate
                double[] ft = new double[hiddenLength];
                MathOp.Axpb(Wf1, U[lid], bf, ft);
                MathOp.Axpy(Wf2, last_ht, ft);
                ft = sigmoid(ft);
                
                //new ct
                double[] ct = new double[hiddenLength];
                MathOp.pointwiseSum(it, ct0, ft, last_ct, ct);
                
                last_ct = ct;
                last_ht = tanh(ct);

				
				if(!train_locs.contains(locs[i])) 
				{
					rank[i*2] = -1;
					rank[i*2+1] = -1;
					continue;
				}

				//forward to hidden state
				double[] next_state = new double[hiddenLength];

                if(last_pos==-1)
                    MathOp.Axpy(W, state, next_state);
                else
                    MathOp.Axpb(W, state, U[locs[last_pos]], next_state);
                last_pos = i;

                //tanh
                next_state=tanh(next_state);
                state = next_state;


				double[] results = new double[loc_Num];
				
				MathOp.Axpy(V2, R[uid], results);
				MathOp.Axpy(V, state, results);
				MathOp.Axpy(V3, userRepre, results);
				MathOp.Axpy(NV, NR[uid], results);
				rank[i*2] = get_rank(results,locs[i]);
				if(existLocs.contains(locs[i]))
					rank[i*2+1] = -1;
				else
					rank[i*2+1] = get_rank_new(results,locs[i],existLocs);
			}
			ranks.add(rank);
			userRepre = last_ht;
		}
		return ranks;		
	}
	
	static int get_rank(double[] score, int id)
	{
		double value = score[id];
		int rk = 1;
		for(int i=0;i<score.length;i++)
			if(score[i]>value)
				++rk;
		return rk;
	}
	
	static int get_rank_new(double[] score, int id, Set<Integer> existLocs)
	{
		double value = score[id];
		int rk = 1;
		for(int i=0;i<score.length;i++)
			if(score[i]>value && !existLocs.contains(i))
				++rk;
		return rk;
	}
	
	public static void clearMatrix(double[][] X)
	{
		for(int i = 0; i < X.length; i++)
		{
			for(int j = 0; j < X[i].length; j++)
			{
				X[i][j] = 0;
			}
		}
	}
	
	public static void clearVector(double[] X)
	{
		for(int i = 0; i < X.length; i++)
		{
			X[i] = 0;
		}
	}
	
	public static void updateMatrix(double[][] X, double[][] XAdaLR, double[][] dX, double learningRate, int batchsize)
	{
		for(int i = 0; i < X.length; i++)
		{
			for(int j = 0; j < X[i].length; j++)
			{
				XAdaLR[i][j] += (dX[i][j] / batchsize) * (dX[i][j] / batchsize);
				if(dX[i][j] !=0)
					X[i][j] += (learningRate / batchsize) * dX[i][j] / Math.sqrt(XAdaLR[i][j]);
			}
		}
	}
	
	public static void updateVector(double[] X, double[] XAdaLR, double[] dX, double learningRate, int batchsize)
	{
		for(int i = 0; i < X.length; i++)
		{
			XAdaLR[i] += (dX[i] / batchsize) * (dX[i]/ batchsize);
			if(dX[i] !=0)
				X[i] += (learningRate / batchsize) * dX[i] / Math.sqrt(XAdaLR[i]);
		}
	}
	
	static public double[] tanh(double[] input)
	{
		double[] output = new double[input.length];
		for (int i = 0; i < input.length; ++i)
        {
            if (input[i] > 0)
            {
                double x = Math.exp(-2.0 * 1 * input[i]);

                output[i] = (1.0 - x) / (1.0 + x);
            }
            else
            {
                double x = Math.exp(2.0 * 1 * input[i]);

                output[i] = (x - 1.0) / (x + 1.0);
            }
        }
		return output;		
	}
	
	static public double[] sigmoid(double[] input)
	{
		double[] output = new double[input.length];
		for (int i = 0; i < input.length; i++)
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
		return output;		
	}
	
	static public double[] tanh_backward(double[] outputG, double[] output) {
		double[] inputG = new double[output.length];
		for (int i = 0; i < inputG.length; ++i)
        {
            inputG[i] = (1.0 - output[i] * output[i]) * outputG[i];
        }
		return inputG;
	}
	
	static public double[] sigmoid_backward(double[] outputG, double[] output) {
		double[] inputG = new double[output.length];
		for (int i = 0; i < outputG.length; ++i)
        {
            inputG[i] = outputG[i] * output[i] * (1.0f - output[i]);
        }
		return inputG;
	}
}
