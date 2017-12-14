using System;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using NUnit.Framework;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Layer;
using OpenMined.Syft.Layer.Loss;
using OpenMined.Syft.Model;
using OpenMined.Syft.Tensor;
using UnityEngine;

namespace OpenMined.Tests.Editor.Model
{
    [Category("ModelCPUTests")]
    public class ModelTest
    {
        public SyftController ctrl;

        [OneTimeSetUp]
        public void Init()
        {
            //Init runs once before running test cases.
            ctrl = new SyftController(null);
        }

        [Test]
        public void TestModelCanLearn()
        {
            float[] inputData = {0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1};
            int[] inputShape = {4, 3};
            var inputTensor = new Syft.Tensor.FloatTensor(ctrl, _data: inputData, _shape: inputShape, _autograd: true);

            float[] targetData = {0, 0, 1, 1};
            int[] targetShape = {4, 1};
            var targetTensor = new Syft.Tensor.FloatTensor(ctrl, _data: targetData, _shape: targetShape, _autograd: true);

            var model = new Syft.Layer.Model(
                new Linear(ctrl, 3, 4),
                new Sigmoid(),
                new Linear(ctrl, 4, 1),
                new Sigmoid()
            );

            float currentLoss = 1;

            // train the model
            for (var i = 0; i < 10; ++i)
            {
                var prediction = model.Predict(inputTensor);
                var loss = MSELoss.Value(prediction, targetTensor);
                loss.Backward();

                foreach (var layer in model.Layers)
                {
                    var weight = layer.GetWeights();
                    weight?.Sub(weight.Grad.Transpose(), true);
                }

                currentLoss = loss.Data.Sum();
            }

            Assert.True(Math.Round(currentLoss, 5) <= 0.20936);
        }
    }
}
