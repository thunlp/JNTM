package model.nn_composition;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import dataprepare.Data;
import dataprepare.Funcs;
import duyuNN.*;
import duyuNN.combinedLayer.*;
import evaluationMetric.Metric;

public class UserMain {
	
	List<String> trainSet = new ArrayList<String>(); //used for train/test set separation
	List<String> testSet = new ArrayList<String>();
	
	public DocSimplifiedLSTM123Main( 
				int embeddingLength,
				int locNum,
				int userNum,
				double input_bias,
				double cand_bias,
				double forget_bias,
				String trainFile,
				String testFile,
				String netFile,
				double randomizeBase) throws Exception
	{
		UserLSTM.input_bias = input_bias;
		UserLSTM.cand_bias = cand_bias;
		UserLSTM.forget_bias = forget_bias;


		UserLSTM.initialize(embeddingLength,locNum,userNum);
		Random rnd = new Random();
		UserLSTM.randomize(rnd, -1.0 * randomizeBase,1.0* randomizeBase);
		
		NetworkModel.initialize(embeddingLength, userNum);
		NetworkModel.randomize(rnd, -1.0 * randomizeBase,1.0* randomizeBase);
		NetworkModel.loadNetwork(netFile);

		
		UserLSTM.loadData(trainFile, testFile);

	}
	
	
	
	public void run(
			int roundNum,
			double probThreshold,
			double learningRate,
			int userNum
			) throws Exception
	{
		UserLSTM.probThreshold = probThreshold;
		UserLSTM.learningRate = learningRate;
		NetworkModel.probThreshold = probThreshold;
		NetworkModel.learningRate = learningRate;
		UserLSTM.batchsize = 32;
		int numThread = 4;
		int netRound = 50;
		
		for(int round = 1; round <= roundNum; round++)
		{
		 	//learningRate -= learningRate / roundNum;
			System.out.println("============== running round: " + round + " ===============");
			
			
            for(int netIter=0;netIter<netRound;netIter++)
            {
                //System.out.println(netIter);
                NetworkModel.current = 0;
                NetworkModel.loss = 0;

                List<NetworkModel> nm = new ArrayList<NetworkModel>();
                for(int th=0;th<numThread;th++)
                    nm.add(new NetworkModel());
                for(int th=0;th<numThread;th++)
                    nm.get(th).start();

                //System.out.println(netIter);
                while(true)
                {
                    boolean br = true;
                    for(int th=0;th<numThread;th++)
                        if(nm.get(th).isAlive())
                            br = false;
                    if(br) break;				
                }
                NetworkModel.updateAdaGrad(learningRate, 1);
                NetworkModel.clearGrad();

                double lossN = NetworkModel.loss;
                int lossS = NetworkModel.nonzero;
                if(netIter==netRound-1 || netIter%10==9)
                    System.out.println("running\t "+round
                    + "lossN/lossS = " + String.format("%.4f", lossN)
                    + "\t" + new Date().toLocaleString()+"\n");
            }
			
			Collections.shuffle(UserLSTM.trainData, new Random());
			System.out.println("Finish shuffling training data.");
			
			UserLSTM.loss = 0;
			UserLSTM.num = 0;

			int cnt=0;
			
			for(int i=0;i<UserLSTM.trainData.size();i++)
			{
				++cnt;
				UserLSTM.current_id = i;

				UserLSTM.lstm_forward();
				
				//multi thread
				List<UserLSTM> ul = new ArrayList<UserLSTM>();
				for(int th=0;th<numThread;th++)
					ul.add(new UserLSTM());
				for(int th=0;th<numThread;th++)
					ul.get(th).start();
					
				while(true)
				{
					boolean br = true;
					for(int th=0;th<numThread;th++)
						if(ul.get(th).isAlive())
							br = false;
					if(br)
					{
						//ul.clear();
						break;	
					}
				}
				

				UserLSTM.lstm_backward();

				UserLSTM.userState = new ArrayList<List<double[]>>();
				UserLSTM.inputState= new ArrayList<List<double[]>>();
				UserLSTM.candState= new ArrayList<List<double[]>>();
				UserLSTM.forgetState= new ArrayList<List<double[]>>();
				UserLSTM.hiddenState= new ArrayList<List<double[]>>();
				UserLSTM.current = 0;
				
				if(cnt % UserLSTM.batchsize == 0)
				{
					UserLSTM.updateAdaGrad(learningRate, UserLSTM.batchsize);
					UserLSTM.clearGrad();			
				}
				else if(cnt == UserLSTM.trainData.size())
				{
					UserLSTM.updateAdaGrad(learningRate, cnt % UserLSTM.batchsize);
					UserLSTM.clearGrad();	
				}
					
				double lossV =UserLSTM.loss;
				int lossC = UserLSTM.num;
				if(cnt % 1024 == 0 || cnt == UserLSTM.trainData.size())
				{
					System.out.println("running idxData = " + cnt + "/" + UserLSTM.trainData.size() + "\t "
								+ "lossV/lossC = " + String.format("%.4f", lossV) + "/" + lossC + "\t"
								+ " = " + String.format("%.4f", lossV/lossC)
								+ "\t" + new Date().toLocaleString()+"\n");
				}

			}
			

			System.out.println("============= finish training round: " + round + " ==============");

			if(round % 5 == 1)
			{
				//save parameter history				
				saveMatrix(round+"_U.txt", UserLSTM.U);
				saveMatrix(round+"_W.txt", UserLSTM.W);
				saveMatrix(round+"_V.txt", UserLSTM.V);
				saveVector(round+"_S.txt", UserLSTM.init_state);
				saveMatrix(round+"_V2.txt", UserLSTM.V2);
				saveMatrix(round+"_R.txt", UserLSTM.R);				
				saveMatrix(round+"_NC.txt", NetworkModel.C);
				saveMatrix(round+"_NR.txt", UserLSTM.NR);
                saveMatrix(round+"_V3.txt", UserLSTM.V3);
                saveMatrix(round+"_Wi1.txt", UserLSTM.Wi1);
                saveMatrix(round+"_Wi2.txt", UserLSTM.Wi2);
                saveMatrix(round+"_Wc1.txt", UserLSTM.Wc1);
                saveMatrix(round+"_Wc2.txt", UserLSTM.Wc2);
                saveMatrix(round+"_Wf1.txt", UserLSTM.Wf1);
                saveMatrix(round+"_Wf2.txt", UserLSTM.Wf2);
                saveVector(round+"_bi.txt", UserLSTM.bi);
                saveVector(round+"_bc.txt", UserLSTM.bc);
                saveVector(round+"_bf.txt", UserLSTM.bf);
                saveVector(round+"_C.txt", UserLSTM.init_C);


				predict(round);
			}
		}
	}
	
	public void saveMatrix(String file, double[][] matrix) throws Exception
	{
		File outputFile = new File(file);
		FileOutputStream fos = new FileOutputStream(outputFile);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
		for(int i=0;i<matrix.length;++i)
		{
			for(int j=0;j<matrix[i].length;++j)
				bw.write(matrix[i][j]+" ");
			bw.write("\n");
		}
		bw.close();
		fos.close();
	}
	
	public void saveVector(String file, double[] vector) throws Exception
	{
		File outputFile = new File(file);
		FileOutputStream fos = new FileOutputStream(outputFile);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
		for(int i=0;i<vector.length;++i)
		{
			bw.write(vector[i]+" ");
		}
		bw.write("\n");
		bw.close();
		fos.close();
	}
	
	public void predict(int round) throws Exception
	{
		System.out.println("=========== predicting round: " + round + " ===============");
		
		
		//prediction
		int hit1 = 0;
		int hit5 = 0;
		int hit10 = 0;
		int total = 0;
		
		int hit1_new = 0;
		int hit5_new = 0;
		int hit10_new = 0;
		int total_new = 0;
		
		for(int idxData = 0; idxData < UserLSTM.testData.size(); idxData++)
		{
			UserLSTM.userState= new ArrayList<List<double[]>>();
			UserLSTM.inputState= new ArrayList<List<double[]>>();
			UserLSTM.candState= new ArrayList<List<double[]>>();
			UserLSTM.forgetState= new ArrayList<List<double[]>>();
			UserLSTM.hiddenState= new ArrayList<List<double[]>>();
			if(idxData%1000 ==1)
				System.out.println(idxData);
			List<int[]> userdata = UserLSTM.testData.get(idxData);
			//changed
			List<int[]> ranks = UserLSTM.rank_prediction(userdata);
			
			for(int j=0;j<ranks.size();j++)
			{
				for(int i=0;i<ranks.get(j).length;i++)
				{					
					if(i%2==0) ++total;
					if(ranks.get(j)[i]==-1) continue;
					if(i%2==1) ++total_new;
					if(ranks.get(j)[i]==1)
						if(i%2==0)
							++hit1;
						else
							++hit1_new;

					if(ranks.get(j)[i]<=5)
						if(i%2==0)
							++hit5;
						else
							++hit5_new;

					if(ranks.get(j)[i]<=10)
						if(i%2==0)
							++hit10;
						else
							++hit10_new;
					/*if(i==0)
					{
						++total_2;
						if(ranks.get(j)[i]==1)
							++hit1_2;
						if(ranks.get(j)[i]<=5)
							++hit5_2;
						if(ranks.get(j)[i]<=10)
							++hit10_2;
					}*/
				}
			}
		}
		UserLSTM.userState= new ArrayList<List<double[]>>();
		UserLSTM.inputState= new ArrayList<List<double[]>>();
		UserLSTM.candState= new ArrayList<List<double[]>>();
		UserLSTM.forgetState= new ArrayList<List<double[]>>();
		UserLSTM.hiddenState= new ArrayList<List<double[]>>();
		
		System.out.println(1.0*hit1/total);
		System.out.println(1.0*hit5/total);
		System.out.println(1.0*hit10/total);
		System.out.println(1.0*hit1_new/total_new);
		System.out.println(1.0*hit5_new/total_new);
		System.out.println(1.0*hit10_new/total_new);
		System.out.println("============== finish predicting =================");
	}
	
	public static void main(String[] args) throws Exception
	{		
		HashMap<String, String> argsMap = Funcs.parseArgs(args);
		
		System.out.println("==== begin configuration ====");
		for(String key: argsMap.keySet())
		{
			System.out.println(key + "\t\t" + argsMap.get(key));
		}
		System.out.println("==== end configuration ====");
		
		int embeddingLength = Integer.parseInt(argsMap.get("-embeddingLength"));

		int locNum = Integer.parseInt(argsMap.get("-locNum"));
		int userNum = Integer.parseInt(argsMap.get("-userNum"));
		
		int roundNum = Integer.parseInt(argsMap.get("-roundNum"));
		double probThreshold = Double.parseDouble(argsMap.get("-probThreshold"));
		double learningRate = Double.parseDouble(argsMap.get("-learningRate"));
		double randomizeBase = Double.parseDouble(argsMap.get("-randomizeBase"));
		
		double input_bias = Double.parseDouble(argsMap.get("-input_bias"));
		double cand_bias = Double.parseDouble(argsMap.get("-cand_bias"));
		double forget_bias = Double.parseDouble(argsMap.get("-forget_bias"));
		
		String trainFile = argsMap.get("-trainFile");
		String testFile  = argsMap.get("-testFile");
		String netFile  = argsMap.get("-netFile");
		
		UserMain main = new UserMain(
				embeddingLength,
				locNum,
				userNum,
				input_bias,
				cand_bias,
				forget_bias,
				trainFile, 
				testFile,
				netFile,
				randomizeBase);
		
		main.run(roundNum, 
				probThreshold, 
				learningRate,
				userNum);

	}
}
