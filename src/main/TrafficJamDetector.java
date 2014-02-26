package main;

import java.io.File;
import java.io.IOException;

import tutorials.TutorialData;
import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.bayes.NaiveBayesClassifier;
import net.sf.javaml.clustering.Clusterer;
import net.sf.javaml.clustering.KMeans;
import net.sf.javaml.clustering.evaluation.AICScore;
import net.sf.javaml.clustering.evaluation.BICScore;
import net.sf.javaml.clustering.evaluation.ClusterEvaluation;
import net.sf.javaml.clustering.evaluation.SumOfSquaredErrors;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.filter.discretize.EqualWidthBinning;
import net.sf.javaml.tools.data.FileHandler;

public class TrafficJamDetector {


	
	private static void kmeans(String file) throws IOException{
		Dataset data = FileHandler.loadDataset(new File(file), 18, "	");
		
		Clusterer km = new KMeans(3);
		Dataset[] clusters = km.cluster(data);
    	System.out.println("Cluster count: " + clusters.length);
    	
    	ClusterEvaluation aic = new AICScore();
        ClusterEvaluation bic = new BICScore();
        ClusterEvaluation sse = new SumOfSquaredErrors();

        double aicScore = aic.score(clusters);
        double bicScore = bic.score(clusters);
        double sseScore = sse.score(clusters);
        
        System.out.println("AIC score: " + aicScore);
        System.out.println("BIC score: " + bicScore);
        System.out.println("Sum of squared errors: " + sseScore);

	}
	
	
	private static void naiveBayes() throws IOException{
		/* Load a data set */
		Dataset data = TutorialData.IRIS.load();

		/* Discretize through EqualWidtBinning */
		EqualWidthBinning eb = new EqualWidthBinning(20);
		System.out.println("Start discretisation");
		eb.build(data);
		Dataset ddata = data.copy();
		eb.filter(ddata);

		boolean useLaplace = true;
		boolean useLogs = true;
		Classifier nbc = new NaiveBayesClassifier(useLaplace, useLogs, false);
		nbc.buildClassifier(data);

		Dataset dataForClassification = TutorialData.IRIS.load();

		/* Counters for correct and wrong predictions. */
		int correct = 0, wrong = 0;

		/* Classify all instances and check with the correct class values */
		for (Instance inst : dataForClassification) {
			eb.filter(inst);
			Object predictedClassValue = nbc.classify(inst);
			Object realClassValue = inst.classValue();
			if (predictedClassValue.equals(realClassValue))
				correct++;
			else {
				wrong++;
			}

		}
		System.out.println("correct " + correct);
		System.out.println("incorrect " + wrong);
	}
	
	
	public static void main(String[] args) throws IOException {
		kmeans("pems_splitData_NB.csv");
	}

}
