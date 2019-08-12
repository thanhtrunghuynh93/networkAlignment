package ModelWith2OrderNorm;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import StaticVar.Vars;

public class BasicUnit {


	public static double getabs(String[] dis_1,String[] dis_2)
	{
		double answer=0;
		for(int i=0;i<dis_1.length;i++)
		{
			double p=Double.parseDouble(dis_1[i]);
			double q=Double.parseDouble(dis_2[i]);
			answer+=Math.abs(p-q);
		}
		return answer;
	}
	public static double getabsDouble(double[] dis_1,double[] dis_2)
	{
		double answer=0;
		for(int i=0;i<dis_1.length;i++)
		{
			double p=dis_1[i];
			double q=dis_2[i];
			answer+=Math.abs(p-q);
		}
		return answer;
	}

	public static double getCosin(String[] dis_1,String[] dis_2)
	{
		double frac_1=0;
		double frac_2_a=0;
		double frac_2_b=0;
		for(int i=0;i<dis_1.length;i++)
		{
			double p=Double.parseDouble(dis_1[i]);
			double q=Double.parseDouble(dis_2[i]);
			frac_1+=p*q;
			frac_2_a+=p*p;
			frac_2_b+=q*q;
		}
		double cosin_diver=frac_1/(Math.sqrt(frac_2_a)*Math.sqrt(frac_2_b));
		return cosin_diver;
	}


	public static double getCosinDouble(double[] dis_1,double[] dis_2)
	{
		double frac_1=0;
		double frac_2_a=0;
		double frac_2_b=0;
		for(int i=0;i<dis_1.length;i++)
		{
			double p=dis_1[i];
			double q=dis_2[i];
			frac_1+=p*q;
			frac_2_a+=p*p;
			frac_2_b+=q*q;
		}
		double cosin_diver=frac_1/(Math.sqrt(frac_2_a)*Math.sqrt(frac_2_b));
		return cosin_diver;
	}

	public static HashMap<String,HashSet<String>> getNetwork(String filename) throws IOException
	{
		HashMap<String,HashSet<String>> aminer_map=
				new HashMap<String,HashSet<String>>();
		BufferedReader br_aminer=new BufferedReader(new FileReader(
				filename));
		String aminer_string=br_aminer.readLine();
		int index=0;
		while(aminer_string!=null)
		{
			String[] array=aminer_string.split("\\s+");
			if(aminer_map.containsKey(array[0]))
			{
				aminer_map.get(array[0]).add(array[1]);
			}
			else
			{
				HashSet<String> temp_set=new HashSet<String>();
				temp_set.add(array[1]);
				aminer_map.put(array[0], temp_set);
			}

			index++;
			if(index%1000000==0)
			{
				System.out.println(index);
			}
			aminer_string=br_aminer.readLine();
		}
		br_aminer.close();
		return aminer_map;
	}


	public static HashMap<String,HashSet<String>> getNetworkReverse(String filename) throws IOException
	{
		HashMap<String,HashSet<String>> aminer_map=
				new HashMap<String,HashSet<String>>();
		BufferedReader br_aminer=new BufferedReader(new FileReader(
				filename));
		String aminer_string=br_aminer.readLine();
		int index=0;
		while(aminer_string!=null)
		{
			String[] array=aminer_string.split("\\s+");
			if(aminer_map.containsKey(array[1]))
			{
				aminer_map.get(array[1]).add(array[0]);
			}
			else
			{
				HashSet<String> temp_set=new HashSet<String>();
				temp_set.add(array[0]);
				aminer_map.put(array[1], temp_set);
			}

			index++;
			if(index%1000000==0)
			{
				System.out.println(index);
			}
			aminer_string=br_aminer.readLine();
		}
		br_aminer.close();
		return aminer_map;
	}


	public static HashMap<String,HashSet<String>> getNetworkUndirect(String filename)
			throws IOException
	{
		HashMap<String,HashSet<String>> aminer_map=
				new HashMap<String,HashSet<String>>();
		BufferedReader br_aminer=new BufferedReader(new FileReader(
				filename));
		String aminer_string=br_aminer.readLine();
		int index=0;
		while(aminer_string!=null)
		{
			String[] array=aminer_string.split("\\s+");
			if(aminer_map.containsKey(array[0]))
			{
				aminer_map.get(array[0]).add(array[1]);
			}
			else
			{
				HashSet<String> temp_set=new HashSet<String>();
				temp_set.add(array[1]);
				aminer_map.put(array[0], temp_set);
			}

			if(aminer_map.containsKey(array[1]))
			{
				aminer_map.get(array[1]).add(array[0]);
			}
			else
			{
				HashSet<String> temp_set=new HashSet<String>();
				temp_set.add(array[0]);
				aminer_map.put(array[1], temp_set);
			}

			index++;
			if(index%1000000==0)
			{
				System.out.println(index);
			}
			aminer_string=br_aminer.readLine();
		}
		br_aminer.close();
		return aminer_map;
	}

	public static HashMap<String,String[]> getUserEmbedding(String filename) throws IOException
	{
		HashMap<String,String[]> answer=new HashMap<String,String[]>();
		BufferedReader br=new BufferedReader(new FileReader(filename));
		String temp_string=br.readLine();
		while(temp_string!=null)
		{
			String[] array=temp_string.split("\\s+");
			String name=array[0];
			answer.put(name, array[1].split("\\|"));

			temp_string=br.readLine();
		}
		br.close();
		return answer;
	}

	public static HashMap<String,String[]> getUserEmbeddingTest(String filename) throws IOException
	{
		HashMap<String,String[]> answer=new HashMap<String,String[]>();
		BufferedReader br=new BufferedReader(new FileReader(filename));
		String temp_string=br.readLine();
		while(temp_string!=null)
		{
			String[] array=temp_string.split("\\s+");
			String name=array[0];
			/*if(array[0].indexOf("_lastfm")==-1)
			{
				temp_string=br.readLine();
				continue;
			}*/
			answer.put(name, array[1].split("\\|"));
			temp_string=br.readLine();
		}
		br.close();
		return answer;
	}

	public static BufferedReader readData(String filename) throws IOException
	{
		BufferedReader br=new BufferedReader(new FileReader(filename));
		return br;
	}
	public static BufferedWriter writerData(String filename) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(filename));
		return bw;
	}

	public static ArrayList<Map.Entry<String, Double>> sortDoubleMap
		(HashMap<String,Double> para)
	{
		List<Entry<String, Double>> list =
				new ArrayList<Entry<String, Double>>(para.entrySet());
        Collections.sort(list,new Comparator<Entry<String, Double>>() {
            @Override
            public int compare(Map.Entry<String, Double> entry1,
            		Map.Entry<String, Double> entry2) {
            	if(entry2.getValue()>entry1.getValue())
            	{
            		return 1;
            	}
            	else if(entry2.getValue()<entry1.getValue())
            	{
            		return -1;
            	}
            	else if(entry2.getValue()==entry1.getValue())
            	{
            		return 0;
            	}

            	return 0;

                //return (entry2.getValue()-entry1.getValue());
            }
        });
        //for (Map.Entry<String, Double> entry:list){
        //    System.out.println(entry.getKey()+" : "+hashMap.get(entry.getKey()));
        //}
        return (ArrayList<Entry<String, Double>>) list;
	}

	public static double getCosinFast(double[] dis_1, double[] dis_2) {
		// TODO Auto-generated method stub
		double answer=0;
		for(int i=0;i<dis_1.length;i++)
		{
			answer+=dis_1[i]*dis_2[i];
		}
		return answer;
	}

}
