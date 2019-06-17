package model.nn_composition;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import duyuNN.MathOp;

public class NetworkModel extends Thread{
	static public double[][] C; //context vector
	static public double[][] CAdaLR; 
	static public double[][] NRAdaLR; 
	static public double[][] dC;
	static public double[][] dNR;
	
	static public int hiddenLength;
	static public int user_Num;
	
	static double probThreshold;
	static double learningRate;
	static double loss;
	static int nonzero;
	
	static public void initialize(int embeddingLength,int userNum)
	{
		hiddenLength = embeddingLength;
		user_Num = userNum;
		
		C = new double[userNum][];
		for (int i = 0; i < userNum; ++i)
        {
			C[i] = new double[embeddingLength];
        }
		
		CAdaLR = new double[userNum][];
		for (int i = 0; i < userNum; ++i)
        {
			CAdaLR[i] = new double[embeddingLength];
        }
		
		NRAdaLR = new double[userNum][];
		for (int i = 0; i < userNum; ++i)
        {
			NRAdaLR[i] = new double[embeddingLength];
        }
		
		dC = new double[userNum][];
		for (int i = 0; i < userNum; ++i)
        {
			dC[i] = new double[embeddingLength];
        }
		
		dNR = new double[userNum][];
		for (int i = 0; i < userNum; ++i)
        {
			dNR[i] = new double[embeddingLength];
        }
	}
	
	static public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		for(int i = 0; i < C.length; i++)
		{
			for(int j = 0; j < C[i].length; j++)
			{
				C[i][j] = r.nextFloat() * (max - min) + min;
			}
		}
	}
	
	 //double[][] groundTruth;
	static List<HashMap<Integer, Double>> groundTruth = new ArrayList<HashMap<Integer,Double>>();
	static public void loadNetwork(
			String networkFile) throws Exception
	{
		System.out.println("================ start loading ground truth network ==============");
		for(int i=0;i<UserLSTM.user_Num;i++)
			groundTruth.add(new HashMap<Integer, Double>());

		BufferedReader br = new BufferedReader(new FileReader(networkFile));
		String s = null;
		while((s = br.readLine())!=null)
		{
			String[] stmp = s.split("\\s+");
			//System.out.println(stmp[0]);
			//System.out.println(stmp[1]);
			
			int id1 = (int)Double.parseDouble(stmp[0]);
			int id2 = (int)Double.parseDouble(stmp[1]);
			//double value = Double.parseDouble(stmp[3]);
			groundTruth.get(id1).put(id2, 1.0);
			++nonzero;
		}
		br.close();
		//System.out.println(groundTruth[35][4]);
		//System.out.println(groundTruth[33][2925]);
		//Thread.sleep(3000);
		System.out.println("================ finish loading ==============");
	}
	
	static private Lock lock = new ReentrantLock();
	static int current = 0;
	static public synchronized int getJob() {
		++current;
        return current-1;
    }
	public void run()
	{
		int i;
		while((i=getJob())<user_Num)
		{
//			System.out.println(i);
			double tmploss = 0;
			//neg samples
			List<Integer> neg_samples = new ArrayList<Integer>();
			Random rnd = new Random();
			for(int j=0;j<user_Num;j++)
			{
				if(rnd.nextDouble()<0.01 && !groundTruth.get(i).containsKey(j)) //alert negative sample ratio is 0.01
					neg_samples.add(j);
			}
			
			for(int j=0;j<neg_samples.size();j++)
			{
				int zid = neg_samples.get(j);
				double dtmp=1+Math.exp(-MathOp.dotProduct(C[zid], UserLSTM.NR[i]));		
				tmploss += Math.log(1-1.0/dtmp);
				
				double outputG = -1.0 / dtmp;
				for(int k=0;k<hiddenLength;k++)
				{
	        		dC[zid][k] += outputG*UserLSTM.NR[i][k];
	        		dNR[i][k] += outputG * C[zid][k];
				}
				
			}
			Set<Integer> ukeys=groundTruth.get(i).keySet();
			for(Integer key:ukeys)
			{

				double dtmp=1+Math.exp(-MathOp.dotProduct(C[key], UserLSTM.NR[i]));
				tmploss += Math.log(1.0/dtmp);
				
				double outputG = 1.0-1.0 / dtmp;
				for(int k=0;k<hiddenLength;k++)
				{
	        		dC[key][k] += outputG*UserLSTM.NR[i][k];
	        		dNR[i][k] += outputG * C[key][k];
				}
			}
						
	        lock.lock();
	        loss += tmploss;
	        lock.unlock();
		}
		this.stop();
	}
	
	static public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		for(int i = 0; i < C.length; i++)
		{
			for(int j = 0; j < C[i].length; j++)
			{
				CAdaLR[i][j] += (dC[i][j] / batchsize) * (dC[i][j] / batchsize);
				if(dC[i][j] !=0)
					C[i][j] += (learningRate / batchsize) * dC[i][j] / Math.sqrt(CAdaLR[i][j]);
			}
		}

		for(int i = 0; i < UserLSTM.NR.length; i++)
		{
			for(int j = 0; j < UserLSTM.NR[i].length; j++)
			{
				NRAdaLR[i][j] += (dNR[i][j] / batchsize) * (dNR[i][j] / batchsize);
				if(dNR[i][j] !=0)
				{
					UserLSTM.NR[i][j] += (learningRate / batchsize) * dNR[i][j] / Math.sqrt(NRAdaLR[i][j]);
					if(UserLSTM.NR[i][j]>20) UserLSTM.NR[i][j]=20;
					if(UserLSTM.NR[i][j]<-20) UserLSTM.NR[i][j]=-20;
				}
			}
		}
	}

	static public void clearGrad() {
		// TODO Auto-generated method stub
		for(int i = 0; i < dC.length; i++)
		{
			for(int j = 0; j < dC[i].length; j++)
			{
				dC[i][j] = 0;
			}
		}
		for(int i = 0; i < dNR.length; i++)
		{
			for(int j = 0; j < dNR[i].length; j++)
			{
				dNR[i][j] = 0;
			}
		}
	}
	public void clearAdaGrad() {
		// TODO Auto-generated method stub
		for(int i = 0; i < CAdaLR.length; i++)
		{
			for(int j = 0; j < CAdaLR[i].length; j++)
			{
				CAdaLR[i][j] = 0;
			}
		}
		for(int i = 0; i < NRAdaLR.length; i++)
		{
			for(int j = 0; j < NRAdaLR[i].length; j++)
			{
				NRAdaLR[i][j] = 0;
			}
		}
	}
}