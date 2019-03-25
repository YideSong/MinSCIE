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
 * @author Yide Song
 */
public class Demo {
    public static void main(String args[]) throws IOException, InterruptedException {
        
        
    	
    	// Dependency parsing pipeline initialization
        StanfordCoreNLP parser = CoreNLPUtils.StanfordDepNNParser();
        
        // Input sentence
        String sentence = "Both taxa are produced in abundance by a variety of coniferous plants, and are typical in the Paleogene of the northern UK and Greenland region (Boulter and Manum 1989; Jolley and Whitham 2004; Jolley and Morton 2007), as well as mid-latitude North America (Smith et al., 2007) and Arctic Canada (Greenwood and Basinger, 1993).";
        
        
        
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
