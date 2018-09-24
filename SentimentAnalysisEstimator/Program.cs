using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Training;

namespace SentimentAnalysisEstimator
{
    class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string TrainDataPath => Path.Combine(AppPath, "data", "wikipedia-detox-250-line-data.tsv");
        private static string TestDataPath => Path.Combine(AppPath, "data", "wikipedia-detox-250-line-test.tsv");

        static (Scalar<bool> label, Scalar<string> text) myfunc(TextLoader.Context ctx)
        {
            return (label: ctx.LoadBool(0), text: ctx.LoadText(1));
        }

        static void Main(string[] args)
        {

            // The first thing to do is create an environment, this can be used for exception tracking and logging
            // Eric: dbcontext in EF, provides the context for the job you are runnning. Random seed for determinism.
            var env = new LocalEnvironment();

            // The other thing on the environment is a component catalog, all the transforms and learners which
            // from an API perspective need to worry about its important to learn the learners etc. 
            env.AddListener((src, message) => Console.WriteLine(message));

            // Text loader is used to load data from a text file, with ML.NET we support csv's/tsv's etc. 
            // The text loader takes in as parameters, env and a func delegate which maps the current context row of 
            // the text file into a named tuple representing the columns. You can also pass in additional setting options
            // like hasheader:true in case of our dataset.
            // Is this an anonymus type?
            var reader2 = TextLoader.CreateReader(env, ctx => (label: ctx.LoadBool(0), text: ctx.LoadText(1)), hasHeader: true);
            var reader1 = TextLoader.CreateReader(env, myfunc, hasHeader: true);

            BinaryClassificationContext ctx2 = new BinaryClassificationContext(env);

            //Once a reader has been defined you can use it read the training data using a multifilesource.
            //Everything we have done so far in this API is lazy, so the actual reading will happen when the data is accessed later.
            var traindata = reader2.Read(new MultiFileSource(TrainDataPath));

            //Once a reader has been defined you can use it read the training data using a multifilesource.
            //Everything we have done so far in this API is lazy, so the actual reading will happen when the data is accessed later.
            var testdata = reader2.Read(new MultiFileSource(TestDataPath));

            //This looks more like a sklearn.pipeline. Esimators take in data and output transforms.  
            var est = traindata.MakeNewEstimator().
                      Append(row => (label: row.label,
                                     prediction: ctx2.Trainers.Sdca(row.label, row.text.FeaturizeText()))).
                      Append(row => (label: row.label,
                                     prediction: row.prediction,
                                     predictedLabel: row.prediction.predictedLabel));

            // Estimator.Fit() will try to learn the parameters of the sdca Binary Classifier that fit the data and return a Transformer (model) with the learnt parameter values
            var model = est.Fit(traindata);

            // The model is a transform so you can call transform on it where you can pass test data and it will return label,prediction,predictedlabel in an IDV.  
            var prediction = model.Transform(testdata);

            var metrics = ctx2.Evaluate(prediction, Row => Row.label, Row => Row.prediction);

            // why dynamic? and why prediction function
            var predictionfunction = model.AsDynamic.MakePredictionFunction<SentimentIssue, SentimentPrediction>(env);

            var predicted = predictionfunction.Predict(new SentimentIssue
            {
                text = "foo"
            });
    }



















        //static void Main(string[] args)
        //{
        //    var env = new LocalEnvironment();

        //    BinaryClassificationContext ctx2 = new BinaryClassificationContext(env);

        //    var reader = TextLoader.CreateReader(env, ctx => (
        //            label: ctx.LoadBool(0),
        //            text: ctx.LoadText(2)), hasHeader: true);

        //    var traindata = reader.Read(new MultiFileSource(TrainDataPath));

        //    var est = traindata.MakeNewEstimator()
        //       .Append(row => (
        //                       label: row.label,
        //                       prediction: ctx2.Trainers.Sdca(row.label, row.text.FeaturizeText())))
        //       .Append(row => (predictedlabel: row.prediction.predictedLabel,
        //                       label: row.label,
        //                       prediction: row.prediction));

        //    var model = est.Fit(traindata);

        //    var testdata = reader.Read(new MultiFileSource(TestDataPath));

        //    var predictions = model.Transform(testdata);

        //    var metrics = ctx2.Evaluate(predictions, row => row.label, row => row.prediction);

        //    var predictor = model.AsDynamic.MakePredictionFunction<SentimentIssue, SentimentPrediction>(env);

        //    var prediction = predictor.Predict(new SentimentIssue
        //    {
        //        text = "This is amazing",
        //    });

        //    System.Console.WriteLine(prediction.predictedlabel);

        //    Console.ReadKey();
        //}

        public class SentimentIssue
        {
            public float label { get; set; }
            public string text { get; set; }
        }

        public class SentimentPrediction
        {
            public bool predictedlabel { get; set; }
        }

    }
}
