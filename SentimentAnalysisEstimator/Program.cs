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

            var reader = TextLoader.CreateReader(env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true);

            var traindata = reader.Read(new MultiFileSource(TrainDataPath));
            
            var est = traindata.MakeNewEstimator()
               .Append(row => (
                               label: row.label,
                               prediction: ctx2.Trainers.Sdca(row.label, row.text.FeaturizeText())));

            var model = est.Fit(traindata);


            var testdata = reader.Read(new MultiFileSource(TestDataPath));

            var predictions = model.Transform(testdata);
            
            var metrics = ctx2.Evaluate(predictions, row => row.label, row => row.prediction);

            Console.ReadKey();
         }
    }
}
