package FinalModel;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

public class INEUpdate {
	
	
	
	HashMap<String,Double> vertex;
	public HashMap<String,double[]> answer=new HashMap<String,double[]>();
	HashMap<String,double[]> answer_context=new HashMap<String,double[]>();
	ArrayList<String> source_id; 
	ArrayList<String> target_id;
	ArrayList<Double> edge_weight;
	ArrayList<Long> alias;
	ArrayList<Double> prob;
	double[] emb_vertex;
	double[] emb_context;
	int dimension;
	String[] neg_table;
	double[] sigmoid_table;
	int sigmoid_table_size;
	int SIGMOID_BOUND;
	double init_rho=0.025,rho;
	int num_negative=5;
	int neg_table_size=10000000;
	String input_file;
	String postfix;
	
	public INEUpdate(int dim,String filename,String postfix_string)
	{
		source_id=new ArrayList<String>();
		target_id=new ArrayList<String>();
		edge_weight=new ArrayList<Double>();
		vertex=new HashMap<String,Double>();
		alias=new ArrayList<Long>();
		prob=new ArrayList<Double>();
		sigmoid_table_size=1000;
		dimension=dim;
		SIGMOID_BOUND=6;
		sigmoid_table=new double[sigmoid_table_size];
		input_file=filename;
		postfix=postfix_string;
	}
	
	public void readData(String filename) throws IOException
	{
		BufferedReader br=new BufferedReader(new FileReader(filename));
		String temp_string=br.readLine();
		int count=0;
		while(temp_string!=null)
		{
			String[] array=temp_string.split("\\s+");
			array[0]=array[0]+"_"+postfix;
			array[1]=array[1]+"_"+postfix;
			source_id.add(array[0]);
			target_id.add(array[1]);

			double weight=1;
			edge_weight.add(weight);
			if(vertex.containsKey(array[0]))
			{
				Double temp_weight=vertex.get(array[0])+
						weight;
				vertex.put(array[0], temp_weight);
			}
			else
			{
				vertex.put(array[0], weight);
				//init vector
				emb_vertex=new double[dimension];
				emb_context=new double[dimension];
				for(int i=0;i<dimension;i++)
				{
						emb_vertex[i]=(Math.random()-0.5)/dimension;
						
						if(Double.isInfinite(emb_vertex[i]))
						{
							System.out.println("init infinite");
						}
				}
				answer.put(array[0], emb_vertex);
				answer_context.put(array[0], emb_context);
				
			}
			
			if(vertex.containsKey(array[1]))
			{
				Double temp_weight=vertex.get(array[1])+
						weight;
				vertex.put(array[1], temp_weight);
			}
			else
			{
				
				vertex.put(array[1], weight);
				emb_vertex=new double[dimension];
				emb_context=new double[dimension];
				for(int i=0;i<dimension;i++)
				{
					
						emb_vertex[i]=(Math.random()-0.5)/dimension;
						if(Double.isInfinite(emb_vertex[i]))
						{
							System.out.println("init infinite");
						}
				}
				answer.put(array[1], emb_vertex);
				answer_context.put(array[1], emb_context);
			}
			temp_string=br.readLine();
			if(count%10000==0)
			{
				System.out.println("Reading Edges "+count);
			}
			count++;
		}
		System.out.println("Number of vertex "+vertex.size()+" Number of Edges "+count);
		br.close();
	}
	
	public void initAliasTable()
	{
		ArrayList<Double> norm_prob=new ArrayList<Double>();
		ArrayList<Long> large_block=new ArrayList<Long>();
		ArrayList<Long> small_block=new ArrayList<Long>();
		prob=new ArrayList<Double>(
				Collections.nCopies(edge_weight.size(), 0.0));
		alias=new ArrayList<Long>(
				Collections.nCopies(edge_weight.size(), (long)0));
		double sum=0;
		long cur_small_block,cur_large_block;
		long num_small_block=0,num_large_block=0;
		for(int i=0;i<edge_weight.size();i++)
		{
			sum+=edge_weight.get(i);
		}
		for(int i=0;i<edge_weight.size();i++)
		{
			norm_prob.add(edge_weight.get(i)*edge_weight.size()/sum);
		}
		for(int i=edge_weight.size()-1;i>=0;i--)
		{
			if(norm_prob.get(i)<1)
			{
				small_block.add((long) i);
				num_small_block++;
			}
			else
			{
				large_block.add((long) i);
				num_large_block++;
			}
		}
		while(num_large_block>0&&num_small_block>0){
			cur_small_block=small_block.get((int) --num_small_block);
			cur_large_block=large_block.get((int) --num_large_block);
			prob.set((int) cur_small_block, norm_prob.get((int) cur_small_block));
			alias.set((int) cur_small_block, cur_large_block);
			double large_prob=norm_prob.get((int) cur_large_block); 
			double small_prob=norm_prob.get((int) cur_small_block); 
			norm_prob.set((int) cur_large_block, 
					large_prob+small_prob-1);
			if(norm_prob.get((int) cur_large_block)<1)
			{
				small_block.set((int) num_small_block++,cur_large_block);
			}
			else
			{
				large_block.set((int) num_large_block++, cur_large_block);
			}
			
		}
		while(num_large_block>0)
		{
			long index=large_block.get((int) --num_large_block);
			prob.set((int) index, 1.0);
		}
		while(num_small_block>0)
		{
			long index=small_block.get((int) --num_small_block);
			prob.set((int) index, 1.0);
		}
		norm_prob.clear();
		small_block.clear();
		large_block.clear();
	}
	
	long sampleAnEdge(double rand_value1, double rand_value2)
	{
		long k=(long) (edge_weight.size()*rand_value1);
		return rand_value2<prob.get((int) k)? k:alias.get((int) k);
	}
	
	void initVector()
	{
		emb_vertex=new double[vertex.size()*dimension];
		emb_context=new double[vertex.size()*dimension];
		for(int i=0;i<dimension;i++)
			for(int j=0;j<vertex.size();j++)
			{
				emb_vertex[j*dimension+i]=(Math.random()-0.5)/dimension;
			}
	}
	
	@SuppressWarnings("rawtypes")
	void initNegTable()
	{
		double sum=0,cur_sum=0,por=0;
		int vid=0;
		
		neg_table=new String[neg_table_size];
		double neg_sampling_power=0.75;
		for(String name:vertex.keySet())
		{
			sum+=Math.pow(vertex.get(name), neg_sampling_power);
		}
		
		Iterator iter=vertex.entrySet().iterator();
		Map.Entry entry=(Map.Entry)iter.next();
		for(int i=0;i<neg_table_size;i++)
		{
			String name=(String) entry.getKey();
			if((double)(i+1)/neg_table_size>por)
			{
				cur_sum+=
						Math.pow((double) entry.getValue(), neg_sampling_power);
				por=cur_sum/sum;
				
				if(por==1||por>1)
				{
					neg_table[i]=name;
					continue;
				}
				if(i!=0)
				{
					
					entry=(Entry) iter.next();
					name=(String) entry.getKey();
				}
				
			}
			neg_table[i]=name;
			
		}
		
	}
	
	public void InitSigmoidTable()
	{
		double x;
		for(int i=0;i<sigmoid_table_size;i++)
		{
			x=2*SIGMOID_BOUND*i/sigmoid_table_size-SIGMOID_BOUND;
			sigmoid_table[i]=1/(1+Math.exp(-x));
		}
		
	}
	
	public double FastSigmoid(double x)
	{
		if(x>SIGMOID_BOUND) return 1;
		else if (x<-SIGMOID_BOUND) return 0;
		int k= (int) ((x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2);
		return sigmoid_table[k];
	}
	
	public void Update(double[] vec_u,
			double[] vec_v,
			double[] vec_error,
			int label,
			String source,
			String target,
			HashMap<String,double[]> TwoOrderAnswer,
			HashMap<String,double[]> TwoOrderAnswerContext,
			HashMap<String,String> anchors)
	{
		double x=0, g;
		
		if(anchors.containsKey(source))
		{
			vec_u=TwoOrderAnswer.get(anchors.get(source));
			if(vec_u==null)
			{
				vec_u=TwoOrderAnswer.get(source);
			}
		}
		if(anchors.containsKey(target))
		{
			vec_v=TwoOrderAnswerContext.get(anchors.get(target));
			if(vec_v==null)
			{
				vec_v=TwoOrderAnswerContext.get(target);
			}
		}
		
		for(int i=0;i<dimension;i++) 
		{
			x+=vec_u[i]*vec_v[i];
		}
		g=(label-FastSigmoid(x))*rho;
		
		
		for(int i=0;i<dimension;i++) {
			vec_error[i]+=g*vec_v[i];
		};
		
		if(anchors.containsKey(target))
		{
			double[] temp_vec=
					TwoOrderAnswerContext.get(anchors.get(target));
			if(temp_vec==null)
			{
				for(int i=0;i<dimension;i++)
				{
					vec_v[i]+=g*vec_u[i];
				}
			}
			else
			{
				for(int i=0;i<dimension;i++)
				{
					TwoOrderAnswerContext.get(anchors.get(target))[i]
							+=g*vec_u[i];
				}
			}
		}
		else
		{
			for(int i=0;i<dimension;i++)
			{
				vec_v[i]+=g*vec_u[i];
			}
		}
	}
	
	public void Train(int i,
			int iter_count,
			HashMap<String,double[]> TwoOrderAnswer,
			HashMap<String,double[]> TwoOrderAnswerContext,
			HashMap<String,String> anchors)
	{
		

			double[] vec_error=new double[dimension];
			if(i%1000000==0)
			{
				
				rho=init_rho*(1.0-(double)i/(double)iter_count);
				if(rho<init_rho*0.0001) rho=init_rho*0.0001;
				System.out.println(i+" "+rho);
			}
			double random1=Math.random();
			double random2=Math.random();
			int edge_id=(int) sampleAnEdge(random1,random2);
			String uid_1=source_id.get(edge_id);
			String uid_2=target_id.get(edge_id);
			String target="";
			int label=0;
			for(int d=0;d<num_negative+1;d++)
			{
				if(d==0)
				{
					label=1;
					target=uid_2;
				}
				else
				{
					int neg_index=(int) (neg_table_size*Math.random());
					target=neg_table[neg_index];
					if(uid_1==null||uid_2==null||target==null)
					{
						System.out.println(uid_1);
						System.out.println(uid_2);
						System.out.println(neg_index);
						System.out.println(target);
					}
					if(target.equals(uid_1)||target.equals(uid_2))
					{
						
						d-=1;
						continue;
					}
					label=0;
				}
				Update(answer.get(uid_1),
						answer_context.get(target),
						vec_error,label,
						uid_1,target,
						TwoOrderAnswer,
						TwoOrderAnswerContext,
						anchors);// second order by anchors guide
			}
			if(anchors.containsKey(uid_1))
			{
				double[] vec_u=TwoOrderAnswer.get(anchors.get(uid_1));
				if(vec_u==null)
				{
					vec_u=TwoOrderAnswer.get(uid_1);
					for(int c=0;c<dimension;c++)
					{
						answer.get(uid_1)[c]+=vec_error[c];
					}
				}
				else
				{
					for(int c=0;c<dimension;c++)
					{
						TwoOrderAnswer.get(anchors.get(uid_1))[c]
								+=vec_error[c];
					}
				}
			}
			else
			{
				for(int c=0;c<dimension;c++)
				{
					answer.get(uid_1)[c]+=vec_error[c];
				}
			}
		//return answer;
	}
	
	public void output(String output_filename) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(
				output_filename));
		for(String uid:answer.keySet())
		{
			bw.write(uid+" ");
			double[] vector=answer.get(uid);
			for(int i=0;i<vector.length;i++)
			{
				bw.write(vector[i]+"|");
			}
			bw.write("\n");
		}
		bw.flush();
		bw.close();
	}
	
	public void init() throws IOException
	{
		readData(input_file);
		initAliasTable();
		initNegTable();
		InitSigmoidTable();
	}

}
