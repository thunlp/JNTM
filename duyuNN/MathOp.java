package duyuNN;

public class MathOp {

	public static void xDotApb(double[] x,
			double[][] A,
			double[] b,
			double[] results)
	{
		int row = x.length;
		int col = results.length;
		
		for(int i = 0; i < col; i++)
		{
			results[i] = b[i];
		}
		
		for(int i = 0; i < row; i++)
		{
			for(int j = 0; j < col; j++)
			{
				results[j] += A[i][j] * x[i];
			}
		}
	}
	
	public static void Axpy(double[][] A, 
			double[] x,
			double[] results)
	{
		int row = results.length;
		int col = x.length;
		
		for(int i = 0; i < row; i++)
		{
			for(int j = 0; j < col; j++)
			{
				results[i] += A[i][j] * x[j];
			}
		}
	}
	
	
	public static void Axpb(double[][] A, 
			double[] x,
			double[] b,
			double[] results)
	{
		int row = results.length;
		int col = x.length;
		
		for(int i = 0; i < row; i++)
		{
			results[i] = b[i];
		}
		
		for(int i = 0; i < row; i++)
		{
			for(int j = 0; j < col; j++)
			{
				results[i] += A[i][j] * x[j];
			}
		}
	}
	
	public static void xdotA(double[] x, double[][] A, double[] results)
    {
		int row = x.length;
		int col = results.length;
		
        for (int j = 0; j < col; ++j)
        {
            results[j] = 0;
        }

        for (int i = 0; i < row; ++i)
        {
            for (int j = 0; j < col; ++j)
            {
            	results[j] += x[i] * A[i][j];
            }
        }
    }
	
	public static void addto(double[][] a,double[][] b) //a := a+b
	{
		for(int i=0;i<a.length;i++)
			for(int j=0;j<a[0].length;j++)
				a[i][j] += b[i][j];
	}
	public static double[][] outProduct(double[] a, double[] b)
	{
		double[][] c =  new double[a.length][];
		for(int i=0;i<a.length;i++)
			c[i] = new double[b.length];
		for(int i=0;i<a.length;i++)
			for(int j=0;j<b.length;j++)
				c[i][j] = a[i]*b[j];
		return c;
	}
	
	
	public static void A_add_xTmulty(double[] x, double[] y, double[][] A)
    {
        int row = x.length;
        int col = y.length;

        for (int i = 0; i < row; ++i)
        {
            for (int j = 0; j < col; ++j)
            {
                A[i][j] += x[i] * y[j];
            }
        }
    }
	
	public static double dotProduct(double[] x, double[] y)
	{
		double dotP = 0;
		for(int i = 0; i < x.length; i++)
		{
			dotP += x[i] * y[i];
		}
		return dotP;
	}
	
	public static void pointwiseSum(double[] x, double[] y, double[] x2, double[] y2, double[] result)
	{
		for(int i = 0; i < x.length; i++)
		{
			result[i] = x[i] * y[i] + x2[i] * y2[i];
		}
	}
	
	public static void xAyB(double[] x, double[][] A,double[] y, double[][] B, double[] results) //Ax+By
    {
		int row = x.length;
		int col = results.length;
		
        for (int j = 0; j < col; ++j)
        {
            results[j] = 0;
        }

        for (int i = 0; i < row; ++i)
        {
            for (int j = 0; j < col; ++j)
            {
            	results[j] += x[i] * A[i][j] + y[i] * B[i][j];
            }
        }
    }
	
	public static void sigmoid(double[] input)
	{
		for (int i = 0; i < input.length; i++)
		{
	        if (input[i] > 0)
	        {
	            double x = (float)Math.exp(-1.0 * input[i]);
	
	            input[i] = 1.0f / (1.0f + x);
	        }
	        else
	        {
	            double x = (float)Math.exp(input[i]);
	
	            input[i] = x / (x + 1.0f);
	        }
		}
	}
	
	public static void tanh(double[] input)
	{
		for (int i = 0; i < input.length; ++i)
        {
            if (input[i] > 0)
            {
                double x = Math.exp(-2.0 * 1 * input[i]);

                input[i] = (1.0 - x) / (1.0 + x);
            }
            else
            {
                double x = Math.exp(2.0 * 1 * input[i]);

                input[i] = (x - 1.0) / (x + 1.0);
            }
        }
	}
	
	public static double[] sum(double[] a,double[] b) //a_i+b_i
	{
		double[] c = new double[a.length];
		for(int i=0;i<a.length;i++)
			c[i] = a[i]+b[i];
		return c;
	}
	
	public static double[] product(double[] a,double[] b) //a_i+b_i
	{
		double[] c = new double[a.length];
		for(int i=0;i<a.length;i++)
			c[i] = a[i]*b[i];
		return c;
	}
}
