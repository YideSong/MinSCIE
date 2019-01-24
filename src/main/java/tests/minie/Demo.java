package tests.minie;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.python.util.PythonInterpreter;

import de.uni_mannheim.minie.MinIE;
import de.uni_mannheim.minie.annotation.AnnotatedProposition;
import de.uni_mannheim.utils.coreNLP.CoreNLPUtils;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;



/**
 * @author Kiril Gashteovski
 */
public class Demo {
    public static void main(String args[]) throws IOException, InterruptedException {
        
//        String exe = "python";
//        String command = "C:/Users/songi/PycharmProjects/Master_Thesis/test.py";
//        String num1 = "Yide doesn't believe that the hero Batman was not born in foggy Gotham City.";
//        String[] cmdArr = new String[] {exe, command, num1};
//        Process pr = Runtime.getRuntime().exec(cmdArr);
//        
//        BufferedReader in = new BufferedReader(new InputStreamReader(pr.getInputStream()));
//        String line;
//        while ((line = in.readLine()) != null) {
//        	System.out.println(line);
//        }
//        in.close();
//        pr.waitFor();
        
    	
    	// Dependency parsing pipeline initialization
        StanfordCoreNLP parser = CoreNLPUtils.StanfordDepNNParser();
        
        // Input sentence
        String sentence = "Both taxa are produced in abundance by a variety of coniferous plants, and are typical in the Paleogene of the northern UK and Greenland region (Boulter and Manum 1989; Jolley and Whitham 2004; Jolley and Morton 2007), as well as mid-latitude North America (Smith et al., 2007) and Arctic Canada (Greenwood and Basinger, 1993).";
//        // Parse the sentence with CoreNLP
//        SemanticGraph sg = CoreNLPUtils.parse(parser, sentence);
        
        
        
        // Generate the extractions (With SAFE mode)
        MinIE minie = new MinIE(sentence, parser, MinIE.Mode.SAFE);
        System.out.println("New Sentence: " + minie.getNewSentence());
        
        // Print the extractions
        System.out.println("\nInput sentence: " + sentence);
        System.out.println("=============================");
        System.out.println("Extractions:");
        for (AnnotatedProposition ap: minie.getPropositions()) {
            System.out.println("\tTriple: " + ap.getTripleAsString());
            System.out.print("\tFactuality: " + ap.getFactualityAsString());
            if(ap.getCitePolarity() != null && ap.getCitePurpose() != null){
            	System.out.print("\tCite: " + ap.getCiteAsString());
            }
            if (ap.getAttribution().getAttributionPhrase() != null) 
                System.out.print("\tAttribution: " + ap.getAttribution().toStringCompact());
            else
                System.out.print("\tAttribution: NONE");
            System.out.println("\n\t----------");
        }
        
        System.out.println("\n\nDONE!");
    }
}
