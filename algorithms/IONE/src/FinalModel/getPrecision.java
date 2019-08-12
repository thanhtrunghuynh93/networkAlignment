package FinalModel;

import ModelWith2OrderNorm.BasicUnit;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class getPrecision {

//	int fold_train;

    //	String gtFile;
    String testFile;
    String srcEmbedFile;
    String trgEmbedFile;
    String srcPostfix, trgPostfix;
    String tempDir; // use to write file

//	public getPrecision(int i)
//	{
//		this.fold_train=i;
//	}

    public getPrecision(String testFile,
                        String srcEmbedFile, String trgEmbedFile,
                        String srcPostfix, String trgPostfix,
                        String tempDir) {
//        this.gtFile=gtFile;
        this.testFile = testFile;
        this.srcEmbedFile = srcEmbedFile;
        this.trgEmbedFile = trgEmbedFile;
        this.srcPostfix = srcPostfix;
        this.trgPostfix = trgPostfix;
        this.tempDir = tempDir;
    }


    public HashMap<String, double[]> getUserEmbeddingTest(String filename) throws IOException {
        HashMap<String, double[]> answer = new HashMap<String, double[]>();
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String temp_string = br.readLine();
        while (temp_string != null) {
            String[] array = temp_string.split("\\s+");
            String name = array[0];
            String[] string_embeddings = array[1].split("\\|");
            double[] temp_embedding = new double[string_embeddings.length];
            double total = 0;
            for (int i = 0; i < string_embeddings.length; i++) {
                double double_tempweight = Double.parseDouble(string_embeddings[i]);
                total += double_tempweight *
                        double_tempweight;
                temp_embedding[i] = double_tempweight;
            }
            double norm_total = Math.sqrt(total);
            for (int i = 0; i < temp_embedding.length; i++) {
                temp_embedding[i] /= norm_total;
            }
            answer.put(name, temp_embedding);
            temp_string = br.readLine();
        }
        br.close();
        return answer;
    }


    public HashSet<String> getAnchors() throws IOException {
        HashSet<String> anchors_set = new HashSet<String>();
//		String anchors_file=
//				Vars.twitter_dir+"/twitter_foursquare_groundtruth/groundtruth."+this.fold_train+".foldtrain.test";
//		BufferedReader br=BasicUnit.readData(anchors_file);
        BufferedReader br = BasicUnit.readData(testFile);
        String temp_string = br.readLine();

        while (temp_string != null) {
            anchors_set.add(temp_string);
            temp_string = br.readLine();
        }
        br.close();
        return anchors_set;
    }

    public TreeMap<Integer, Integer> getTopSimilarity(int k) throws IOException {
        TreeMap<Integer, Integer> answer = new TreeMap<Integer, Integer>();
        for (int i = 0; i < k; i++) {
            answer.put(i + 1, 0);
        }
        HashSet<String> anchors = getAnchors();
        String myspace = srcEmbedFile;

        String lastfm = trgEmbedFile;
        HashMap<String, double[]> myspace_embedding = getUserEmbeddingTest(myspace);
        HashMap<String, double[]> lastfm_embedding = getUserEmbeddingTest(lastfm);

        for (String uid : myspace_embedding.keySet()) {

            if (!anchors.contains(uid.replace("_" + srcPostfix, ""))) {
                // node do not map to any other nodes.
                continue;
            }

            HashMap<String, Double> temp_answer = new HashMap<String, Double>();
            double[] embedding_1 = myspace_embedding.get(uid);
            for (String last_id : lastfm_embedding.keySet()) {
                double[] embedding_2 = lastfm_embedding.get(last_id);
                double cosin_value = BasicUnit.getCosinFast(embedding_1, embedding_2);
                temp_answer.put(last_id, cosin_value);
            }
            String lastfm_name = uid.replace("_" + srcPostfix, "");
            ArrayList<Map.Entry<String, Double>> temp_answer_list = BasicUnit.sortDoubleMap(temp_answer);
            for (int i = 0; i < temp_answer_list.size(); i++) {
                if (lastfm_name.equals(temp_answer_list.get(i).getKey().replace("_" + trgPostfix, ""))) {
                    for (int j : answer.keySet()) {
                        if (i < j) {
                            int temp_count = answer.get(j) + 1;
                            answer.put(j, temp_count);
                        }
                    }
                }
                if (i == k - 1) {
                    break;
                }
            }
        }
        return answer;
    }

    public TreeMap<Integer, Integer> getTopSimilarityReverse(int k) throws IOException {
        TreeMap<Integer, Integer> answer = new TreeMap<Integer, Integer>();
        for (int i = 0; i < k; i++) {
            answer.put(i + 1, 0);
        }
        HashSet<String> anchors = getAnchors();
        String myspace = trgEmbedFile;
        String lastfm = srcEmbedFile;
        HashMap<String, double[]> myspace_embedding = getUserEmbeddingTest(myspace);
        HashMap<String, double[]> lastfm_embedding = getUserEmbeddingTest(lastfm);
        for (String uid : myspace_embedding.keySet()) {
            if (!anchors.contains(uid.replace("_" + trgPostfix, ""))) {
                continue;
            }
            HashMap<String, Double> temp_answer = new HashMap<String, Double>();
            double[] embedding_1 = myspace_embedding.get(uid);
            ArrayList<String> user_set = new ArrayList<String>();
            for (String last_id : lastfm_embedding.keySet()) {
                double[] embedding_2 = lastfm_embedding.get(last_id);
                double cosin_value = BasicUnit.getCosinFast(embedding_1, embedding_2);
                temp_answer.put(last_id, cosin_value);
            }
            ArrayList<Map.Entry<String, Double>> temp_answer_list = BasicUnit.sortDoubleMap(temp_answer);
            String lastfm_name = uid.replace("_" + trgPostfix, "");
            int index = 0;
            for (int i = 0; i < temp_answer_list.size(); i++) {
                if (lastfm_name.equals(temp_answer_list.get(i).getKey().replace("_" + srcPostfix, ""))) {
                    for (int j : answer.keySet()) {
                        if (i < j) {
                            int temp_count = answer.get(j) + 1;
                            answer.put(j, temp_count);
                        }
                    }
                }
                if (i == k - 1) {
                    break;
                }
            }
        }
        return answer;

    }

    public void getFinalAnswer() throws IOException {
        HashSet<String> anchors = getAnchors();
        double total = anchors.size() * 2;
        TreeMap<Integer, Integer> answer_1 = getTopSimilarity(30);
        TreeMap<Integer, Integer> answer_2 = getTopSimilarityReverse(30);
        System.out.println("Map result:");
        for (int key : answer_1.keySet()) {
            double right = answer_1.get(key) + answer_2.get(key);
            System.out.print(right / total + " ");
        }

        System.out.println();

    }

    public static void main(String[] args) throws IOException {
        getPrecision test = new getPrecision(
                "AcrossNetworkEmbeddingData/twitter_foursquare_groundtruth/groundtruth.9.foldtrain.test",
                "AcrossNetworkEmbeddingData/twitter/embeddings/twitter.embedding.update.2SameAnchor.twodirectionContext.100_dim.10000000",
                "AcrossNetworkEmbeddingData/foursquare/embeddings/foursquare.embedding.update.2SameAnchor.twodirectionContext.100_dim.10000000",
                "twitter",
                "foursquare",
                "temp"
        );
        test.getFinalAnswer();
    }

}
