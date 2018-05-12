using System;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;

namespace SentimentAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            const string _dataPath = @"..\..\data\sentiment labelled sentences\imdb_labelled.txt";
            const string _testDataPath = @"..\..\data\sentiment labelled sentences\yelp_labelled.txt";

            var model = TrainAndPredict(_dataPath, _testDataPath);
            Evaluate(model, _testDataPath);

            Console.Read();
        }
        public static PredictionModel<SentimentData, SentimentPrediction> TrainAndPredict(string _dataPath,string _testDataPath)
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<SentimentData>(_dataPath, useHeader: false, separator: "tab"));
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 150, NumTrees = 25, MinDocumentsInLeafs = 5 });

            PredictionModel<SentimentData, SentimentPrediction> model =
                pipeline.Train<SentimentData, SentimentPrediction>();

            IEnumerable<SentimentData> sentiments = new[]
             {
                new SentimentData
                {
                    SentimentText = "Contoso's 11 is a wonderful experience",
                    Sentiment = 0
                },
                new SentimentData
                {
                    SentimentText = "The acting in this movie is very bad",
                    Sentiment = 0
                },
                new SentimentData
                {
                    SentimentText = "Joe versus the Volcano Coffee Company is a great film.",
                    Sentiment = 0
                }
            };

            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);

            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");

            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));

            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive" : "Negative")}");
            }
            Console.WriteLine();

            return model;
        }

        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model,string _testDataPath)
        {
            var testData = new TextLoader<SentimentData>(_testDataPath, useHeader: false, separator: "tab");
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }
    }
}
