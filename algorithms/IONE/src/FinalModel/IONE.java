package FinalModel;

import ModelWith2OrderNorm.BasicUnit;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.text.DecimalFormat;
import java.util.Set;
import java.util.Random;

public class IONE {

    //network x as the foursquare, network y as the twitter

    String train_file, output_filename_networkx, output_filename_networky;
    String networkx_file, networky_file;
    String postfix1, postfix2;
    int total_iter, dim;
    String tempDir; //use to write file
    Random random = new Random();

    //    embeddings
    HashMap<String, double[]> srcEmbed;
    HashMap<String, double[]> trgEmbed;

    public IONE(String networkx_file,
                String networky_file,
                String train_file,
                String postfix1,
                String postfix2,
                int total_iter,
                int dim,
                String tempDir,
                int seed) {
        this.train_file = train_file;
        this.networkx_file = networkx_file;
        this.networky_file = networky_file;
        this.postfix1 = postfix1;
        this.postfix2 = postfix2;
        this.output_filename_networkx = parentDir(this.networkx_file) + "/embeddings/embedding";
        this.output_filename_networky = parentDir(this.networky_file) + "/embeddings/embedding";
        this.total_iter = total_iter;
        this.dim = dim;
        this.tempDir = tempDir;
//        precision = new getPrecision(
//                this.output_filename_networkx,
//                this.output_filename_networky,
//                postfix1,
//                postfix2,
//                tempDir
//        );
        this.random.setSeed(seed);
    }

    private String parentDir(String path) {
        String elms[] = path.split("[\\/]");
        elms[elms.length - 1] = "";
        return String.join("/", elms);
    }

    public HashMap<String, String> getNetworkAnchors(String postfix_1, String postfix_2) throws IOException {
        HashMap<String, String> answer_map = new HashMap<String, String>();
        BufferedReader br = BasicUnit.readData(train_file);
        String temp_string = br.readLine();
        while (temp_string != null) {
            answer_map.put(temp_string + postfix_1, temp_string + postfix_2);
            temp_string = br.readLine();
        }
        return answer_map;
    }

    public void Train() throws IOException {
        IONEUpdate twoOrder_x =
                new IONEUpdate(dim, networkx_file, postfix1, random);
        twoOrder_x.init();

        IONEUpdate twoOrder_y =
                new IONEUpdate(dim, networky_file, postfix2, random);
        twoOrder_y.init();

        HashMap<String, String> lastfm_anchor = getNetworkAnchors("_" + postfix1, "_" + postfix2);
        HashMap<String, String> myspace_anchor = getNetworkAnchors("_" + postfix2, "_" + postfix1);
        System.out.println(lastfm_anchor.size());
        System.out.println(myspace_anchor.size());

        for (int i = 0; i < total_iter; i++) {
            //TwoOrder.Train(i, total_iter);
            twoOrder_x.Train(i, total_iter, twoOrder_x.answer,
                    twoOrder_x.answer_context_input,
                    twoOrder_x.answer_context_output,
                    lastfm_anchor);
            twoOrder_y.Train(i, total_iter, twoOrder_x.answer,
                    twoOrder_x.answer_context_input,
                    twoOrder_x.answer_context_output,
                    myspace_anchor);
//            if ((i + 1) % 10000000 == 0) {
//                twoOrder_x.output(output_filename_networkx);
//                twoOrder_y.output(output_filename_networky);
//            }
        }
//        twoOrder_x.output(output_filename_networkx);
//        twoOrder_y.output(output_filename_networky);
        this.srcEmbed = twoOrder_x.answer;
        this.trgEmbed = twoOrder_y.answer;
    }

    public void align() throws IOException {
        Train();
        createAlignmentMatrix();
    }

    private void createAlignmentMatrix() throws IOException {
        System.gc();
        try {
            PrintWriter pr = new PrintWriter(this.tempDir + "/alignment_matrix");
            double[] embedding_2, embedding_1;
            double cosin_value;
            String printContent = "";
            DecimalFormat nineDForm = new DecimalFormat("#.#########");
            String trgIds[] = new String[trgEmbed.keySet().size()];

            int i = 0;
            for (String lastId: trgEmbed.keySet()) {
                trgIds[i] = lastId.replace("_trg", "");
                i++;
            }

            pr.write(","+String.join(",",trgIds)+"\n");
            for (String uid : srcEmbed.keySet()) {
                printContent = uid.replace("_src","");
                embedding_1 = srcEmbed.get(uid);
                String cosins[] = new String[trgIds.length];
                i = 0;
                for (String last_id : trgIds) {
                    embedding_2 = trgEmbed.get(last_id+"_trg");
                    cosin_value = BasicUnit.getCosinFast(embedding_1, embedding_2);
                    cosins[i] = String.valueOf(Double.valueOf(nineDForm.format(cosin_value)));
                    i++;
                }
                printContent+=","+String.join(",", cosins);
                pr.write(printContent+"\n");
            }
            pr.close();
        } catch (Exception err) {
            err.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException {

        String train_file = "../../temp/groundtruth/groundtruth.train";
        String networkx_file = "../../temp/src/src.edgelist";
        String networky_file = "../../temp/trg/trg.edgelist";
        String postfix1 = "src";
        String postfix2 = "trg";
        int total_iter = 10;
        int dim = 100;
        String tempDir = "temp";
        int seed = 123;
        if (args.length >= 5) {
            networkx_file = args[0];
            networky_file = args[1];
            train_file = args[2];
            postfix1 = args[3];
            postfix2 = args[4];
        }
        if (args.length >= 6) total_iter = Integer.parseInt(args[5]);
        if (args.length >= 7) dim = Integer.parseInt(args[6]);
        if (args.length >= 8) tempDir = args[7];
        if (args.length >= 9) seed = Integer.parseInt(args[8]);

        IONE test = new IONE(
                networkx_file,
                networky_file,
                train_file,
                postfix1,
                postfix2,
                total_iter,
                dim,
                tempDir,
                seed
        );
        test.align();
    }


}
