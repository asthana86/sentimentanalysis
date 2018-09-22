using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
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

        static void Main(string[] args)
        {
            var env = new LocalEnvironment();
            
            BinaryClassificationContext ctx2 = new BinaryClassificationContext(env); 

            var data = TextLoader.CreateReader(env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Read(new MultiFileSource(TrainDataPath));
            
            var testdata = TextLoader.CreateReader(env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Read(new MultiFileSource(TestDataPath));

            var est = data.MakeNewEstimator()
               .Append(row => (
                               label: row.label,
                               features: row.text.FeaturizeText()))
               .Append(row => (
                               label: row.label, 
                               prediction: ctx2.Trainers.Sdca(row.label, row.features)));

            var model = est.Fit(data);

            var predictions = model.Transform(testdata);

            var metrics = ctx2.Evaluate(predictions, row => row.label, row => row.prediction);

            Console.ReadKey();
         }
    }
}
