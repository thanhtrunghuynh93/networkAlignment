package FinalModel;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashMap;

import ModelWith2OrderNorm.BasicUnit;
import StaticVar.Vars;



public class IONES {
	
	int training_ratio;
	public IONES(int train_ratio)
	{
		this.training_ratio=train_ratio;
	}
	
	public HashMap<String,user_sim> getPredictTwitterAnchors() throws IOException
	{
		HashMap<String,user_sim> answer_map=new HashMap<String,user_sim>();
		String file_name=Vars.twitter_dir+"twitter_foursquare_groundtruth/groundtruth.test.linkp."+training_ratio;
		BufferedReader br=BasicUnit.readData(file_name);
		String temp_string=br.readLine();
		while(temp_string!=null)
		{
			user_sim temp_obj=new user_sim();
			String[] array=temp_string.split("\\s+");
			temp_obj.user=array[1]+"_foursquare";
			temp_obj.sim=Double.parseDouble(array[2]);
			answer_map.put(array[0]+"_twitter",temp_obj);
			temp_string=br.readLine();
		}
		return answer_map;
	}
	
	public HashMap<String,user_sim> getPredictFoursquareAnchors() throws IOException
	{
		HashMap<String,user_sim> answer_map=new HashMap<String,user_sim>();
		String file_name=Vars.twitter_dir+"twitter_foursquare_groundtruth/groundtruth.test.linkp."+this.training_ratio;
		BufferedReader br=BasicUnit.readData(file_name);
		String temp_string=br.readLine();
		while(temp_string!=null)
		{
			user_sim temp_obj=new user_sim();
			String[] array=temp_string.split("\\s+");
			temp_obj.user=array[0]+"_twitter";
			temp_obj.sim=Double.parseDouble(array[2]);
			answer_map.put(array[1]+"_foursquare",temp_obj);
			temp_string=br.readLine();
		}
		return answer_map;
	}
	
	public void RealignPredictedAnchors(HashMap<String,double[]> TwitterAnswer,
			HashMap<String,double[]> TwitterInputContext,
			HashMap<String,double[]> TwitterOutputContext,
			HashMap<String,double[]> FoursquareAnswer,
			HashMap<String,double[]> FoursquareInputContext,
			HashMap<String,double[]> FoursquareOutputContext,
			HashMap<String,user_sim> TwitterAnchors
			)
	{
		for(String key:TwitterAnswer.keySet())
		{
			if(TwitterAnchors.containsKey(key))
			{
				System.out.println("contains");
				String temp_foursquare=TwitterAnchors.get(key).user;
				double[] temp=TwitterAnswer.get(key);
				for(int i=0;i<temp.length;i++)
				{
					TwitterAnswer.get(key)[i]=FoursquareAnswer.get(temp_foursquare)[i];
					TwitterInputContext.get(key)[i]=FoursquareInputContext.get(temp_foursquare)[i];
					TwitterOutputContext.get(key)[i]=FoursquareOutputContext.get(temp_foursquare)[i];
				}
			}
				
		}
	}
	
	public HashMap<String,String> getLastFMAnchors() throws IOException
	{
		HashMap<String,String> answer_map=new HashMap<String,String>();
		String file_name=Vars.twitter_dir+"twitter_foursquare_groundtruth/groundtruth.9.foldtrain.train";
		BufferedReader br=BasicUnit.readData(file_name);
		String temp_string=br.readLine();
		while(temp_string!=null)
		{
			answer_map.put(temp_string+"_foursquare", temp_string+"_twitter");
			temp_string=br.readLine();
		}
		return answer_map;
	}
	
	public HashMap<String,String> getMyspaceAnchors() throws IOException
	{
		HashMap<String,String> answer_map=new HashMap<String,String>();
		String file_name=Vars.twitter_dir+"twitter_foursquare_groundtruth/groundtruth.9.foldtrain.train";
		BufferedReader br=BasicUnit.readData(file_name);
		String temp_string=br.readLine();
		while(temp_string!=null)
		{
			answer_map.put(temp_string+"_twitter", temp_string+"_foursquare");
			temp_string=br.readLine();
		}
		return answer_map;
	}
	public void Train(int total_iter,String fileInterop,int dim) throws IOException
	{		
		String last_file=
				StaticVar.Vars.twitter_dir+"/foursquare/following"+fileInterop;
		String ouput_filename_last=
				StaticVar.Vars.twitter_dir+
				"/foursquare/embeddings/foursquare.embedding.update.2SameAnchor.twodirectionContext.Predict.imb."
						+training_ratio+"."+fileInterop;
		
		String myspace_file=
				StaticVar.Vars.twitter_dir+"/twitter/following"+fileInterop;
		String ouput_filename_myspace=
				StaticVar.Vars.twitter_dir+
				"/twitter/embeddings/twitter.embedding.update.2SameAnchor.twodirectionContext.Predict.imb."
						+training_ratio+"."+fileInterop;

		IONESUpdate twoOrder_last=
				new IONESUpdate(dim,last_file,"foursquare");
		twoOrder_last.init();
		
		IONESUpdate twoOrder_myspace=
				new IONESUpdate(dim,myspace_file,"twitter");
		twoOrder_myspace.init();
		
		HashMap<String,String> lastfm_anchor=getLastFMAnchors();
		HashMap<String,String> myspace_anchor=getMyspaceAnchors();
		HashMap<String,user_sim> foursquare_predicted_anchor=this.getPredictFoursquareAnchors();
		HashMap<String,user_sim> twitter_predicted_anchor=this.getPredictTwitterAnchors();

		System.out.println(lastfm_anchor.size());
		System.out.println(myspace_anchor.size());
		
		for(int i=0;i<total_iter;i++)
		{
			twoOrder_last.Train(i, total_iter, twoOrder_last.answer,
					twoOrder_last.answer_context_input,
					twoOrder_last.answer_context_output,
					twoOrder_myspace.answer,
					twoOrder_myspace.answer_context_input,
					twoOrder_myspace.answer_context_output,
					lastfm_anchor,
					foursquare_predicted_anchor);
			twoOrder_myspace.Train(i, total_iter, twoOrder_last.answer,
					twoOrder_last.answer_context_input,
					twoOrder_last.answer_context_output,
					twoOrder_last.answer,
					twoOrder_last.answer_context_input,
					twoOrder_last.answer_context_output,
					myspace_anchor,
					twitter_predicted_anchor);
			if((i+1)%10000000==0)
			{
				twoOrder_last.output(ouput_filename_last+"."+dim+"_dim"+"."+(i+1));
				twoOrder_myspace.output(ouput_filename_myspace+"."+dim+"_dim"+"."+(i+1));
			}
		}
	}
	public static void main(String[] args) throws IOException
	{
		for(int i=9;i<10;i++)
		{
			IONES test=
					new IONES(1);
			test.Train(10000000, "", 100);
		}
	}
	

}
