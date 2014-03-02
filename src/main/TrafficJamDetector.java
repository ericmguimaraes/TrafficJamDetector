package main;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import com.sun.org.apache.bcel.internal.util.Class2HTML;

import tutorials.TutorialData;
import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.classification.bayes.NaiveBayesClassifier;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
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

	private static void kmeans(String file, int classPosition, int k,
			boolean escolhaAleatoria, boolean viewClusters, boolean metricas) throws IOException {
		Dataset data = FileHandler.loadDataset(new File(file), classPosition,
				"	");

		System.out.println("------KMeans------");
		
		Clusterer km = null;
		if(escolhaAleatoria){
			System.out.println("Escolhendo centroid aleatorio");
			km = new KMeans(k);
		}else{
			System.out.println("Escolhendo centroid distantes entre si");
			km = new KmeansModified(k);
		}
		System.out.println("------Treinando------");
		
		Dataset[] clusters = km.cluster(data);
		System.out.println("OK");
		System.out.println("Cluster count: " + clusters.length);

		if (viewClusters) {
			for (Dataset dataset : clusters) {
				System.out.println("*********************CLUSTER*******************");
				int count = 0, inicio = 0, instancias = 15; //max=26000
				for (Instance instance : dataset) {
					if (count > inicio)
						System.out.println(instance.toString());
					count++;
					if (count > inicio+instancias)
						break;
				}
			}

		}

		if (metricas) {
			System.out.println("------Avaliando------");

			//ClusterEvaluation aic = new AICScore();
			//ClusterEvaluation bic = new BICScore();
			ClusterEvaluation sse = new SumOfSquaredErrors();

			//double aicScore = aic.score(clusters);
			//double bicScore = bic.score(clusters);
			double sseScore = sse.score(clusters);

			//System.out.println("AIC score: " + aicScore);
			//System.out.println("BIC score: " + bicScore);
			System.out.println("Sum of squared errors: " + sseScore);

		}

	}

	private static void naiveBayes(String file, int classPosition,
			boolean metricas) throws IOException {
		System.out.println("------Naive Bayes------");
		Dataset data = FileHandler.loadDataset(new File(file), classPosition,
				"	");

		/* Discretize through EqualWidtBinning */
//		 EqualWidthBinning eb = new EqualWidthBinning(1000);
//		 System.out.println("Start discretisation");
//		 eb.build(data);
//		 Dataset ddata = data.copy();
//		 eb.filter(ddata);

		System.out.println("------Treinando------");
		boolean useLaplace = true;
		boolean useLogs = true;
		Classifier nbc = new NaiveBayesClassifier(useLaplace, useLogs, false);
		nbc.buildClassifier(data);
		System.out.println("OK");

		classifica(file, classPosition, nbc, metricas);
	}

	public static void knn(String file, int classPosition, int k,
			boolean metricas) throws IOException {
		System.out.println("------" + k + "NN------");

		Dataset data = FileHandler.loadDataset(new File(file), classPosition,
				"	");
		System.out.println("------Treinando------");
		Classifier knn = new KNearestNeighbors(k);
		knn.buildClassifier(data);
		System.out.println("OK");

		classifica(file, classPosition, knn, metricas);
	}

	public static void classifica(String file, int classPosition,
			Classifier classificador, boolean metricas) throws IOException {
		System.out.println("------Classificando------");
		Dataset dataForClassification = FileHandler.loadDataset(new File(file),
				classPosition, "	");
		int correct = 0, wrong = 0;

		for (Instance inst : dataForClassification) {
			Object predictedClassValue = classificador.classify(inst);
			Object realClassValue = inst.classValue();
			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions  " + correct);
		System.out.println("Wrong predictions " + wrong);

		if (metricas) {
			Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(
					classificador, dataForClassification);
			for (Object o : pm.keySet()) {
				System.out.println("------" + o + "------");
				System.out.println("Accuracy : " + pm.get(o).getAccuracy());
				System.out.println("Reccal : " + pm.get(o).getRecall());
				System.out.println("Precision : " + pm.get(o).getPrecision());
			}
		}
	}

	public static void main(String[] args) throws IOException {
		String file = "pems_splitData_NB_4.csv";
		int ClassPosition = 10;
		boolean metricas = true;
		boolean viewClusters = false;
		
		System.out.println("_____________________________");
		
		naiveBayes(file, ClassPosition, metricas);
		
		System.out.println("_____________________________");
		
		kmeans(file, ClassPosition, 3, true, viewClusters, metricas);
		
		System.out.println("_____________________________");
		
		kmeans(file, ClassPosition, 3, false, viewClusters, metricas);
		
		System.out.println("_____________________________");

		knn(file, ClassPosition, 5, metricas);
	}
}
