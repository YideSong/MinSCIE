package tests.minie;

import java.io.*;
import org.python.util.PythonInterpreter;

import de.uni_mannheim.minie.MinIE;
import de.uni_mannheim.minie.annotation.AnnotatedProposition;
import de.uni_mannheim.utils.coreNLP.CoreNLPUtils;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;



/**
 * @author Kiril Gashteovski
 */
public class DetectCitationDemo {
    public static void main(String args[]) throws IOException, InterruptedException {
    	// Dependency parsing pipeline initialization
        StanfordCoreNLP parser = CoreNLPUtils.StanfordDepNNParser();


        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new File("CitationSentences.csv"));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        
        StringBuilder builder = new StringBuilder();
        builder.append("id");
        builder.append(',');
        builder.append("Input Sentence");
        builder.append(',');
        builder.append("New Sentence");
        builder.append('\n');
        
        File file = new File("src/main/SVM_model_for_citation_classification/Evaluation_Data/oa_200randsents.txt"); 
        
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } 
        
        String st; 
        int id = 0;
        try {
            while ((st = br.readLine()) != null){ 
            	//remove "," in the sentence.
            	st = removeId(st);
            	MinIE minie = new MinIE(st, parser, MinIE.Mode.SAFE);
            	String nst = minie.getNewSentence();
            	
            	if(minie.isCitation()==true){
            		System.out.println("This sentence is citation sentence: " + st);
	            	String handleStr=st;
	            	if(st.contains(",")){             
	            	    if(st.contains("\"")){
	            	    	handleStr=st.replace("\"", "\"\"");
	            	    }  
	            	    handleStr="\""+handleStr+"\"";  
	            	}
	            	
	            	String handleStr2=nst;
	            	if(nst.contains(",")){             
	            	    if(nst.contains("\"")){
	            	    	handleStr2=nst.replace("\"", "\"\"");
	            	    }  
	            	    handleStr2="\""+handleStr2+"\"";  
	            	}
	            	System.out.println(handleStr);
	                builder.append(id);
	                builder.append(',');
	                builder.append(handleStr);
	                builder.append(',');
	                builder.append(handleStr2);
	                builder.append(',');
	                builder.append(minie.getCitePolarity());
	                builder.append(',');
	                builder.append(minie.getCitePurpose());
	                builder.append('\n');
	                id++;
	                
	                for (AnnotatedProposition ap: minie.getPropositions()) {
	                	builder.append(',');
	                	builder.append(ap.getTripleAsString());
	                	builder.append('\n');
	                }
	                builder.append('\n');
            	}
            }
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        pw.write('\ufeff');
        pw.write(builder.toString());
        pw.close();
        System.out.println("done!");
        
        //runMinIE();
    }
    
    public static String removeId (String sentence){
    	sentence = sentence.replaceAll("S[0-9A-Z]*\\:[0-9]*", "");
    	return sentence;
    }
    
    public static void runMinIE() {
    	// Dependency parsing pipeline initialization
        StanfordCoreNLP parser = CoreNLPUtils.StanfordDepNNParser();
        
        // Input sentence
        String sentence = "The Joker believes that the hero Batman was not actually born in foggy Gotham City (Walters, 1994).";

        // Generate the extractions (With SAFE mode)
        MinIE minie = new MinIE(sentence, parser, MinIE.Mode.SAFE);
        
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
