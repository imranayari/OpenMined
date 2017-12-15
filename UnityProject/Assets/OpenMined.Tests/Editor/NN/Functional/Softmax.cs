using System;
using System.Linq;
using NUnit.Framework;
using OpenMined.Network.Controllers;

namespace OpenMined.Tests.Editor.NN.Functional
{
    
    [System.ComponentModel.Category("NN.FunctionalCPUTests")]
    public class Softmax
    {
        private SyftController _ctrl;

#pragma warning disable 618
        [TestFixtureSetUp]
#pragma warning restore 618
        
        public void Init()
        {
            //Init runs once before running test cases.
            _ctrl = new SyftController(null);
        }
        
//        [Test]
//        public void SoftmaxSumsToOne()
//        {
//            float[] data = {4.32f, 1.32f, 0.838f, 1.111f, 0.0001f};
//            int[] shape = {5};
//            var tensor = new Syft.Tensor.FloatTensor(_ctrl, _data: data, _shape: shape);
//
//            var softmax = Syft.NN.Functional.Softmax(tensor);
//
//            var total = softmax.Data.Sum();
//            Assert.AreEqual(1, total);
//        }
//
//        [Test]
//        public void AllNumbersAboveZero()
//        {
//            float[] data = {32.23f, -11f, -30f};
//            int[] shape = {3};
//            var tensor = new Syft.Tensor.FloatTensor(_ctrl, _data: data, _shape: shape);
//            var softmax = Syft.NN.Functional.Softmax(tensor);
//
//            for (var i = 0; i < softmax.Size; ++i)
//            {
//                Assert.GreaterOrEqual(softmax[i], 0);
//            }
//        }
//
//        [Test]
//        public void NumberGetsCloseTo1()
//        {
//            float[] data = {30, 1, 1f};
//            int[] shape = {3};
//            var tensor = new Syft.Tensor.FloatTensor(_ctrl, _data: data, _shape: shape);
//            var softmax = Syft.NN.Functional.Softmax(tensor);
//
//            Assert.GreaterOrEqual(softmax[0], 0.99f);
//        }
//
//        [Test]
//        public void SoftmaxMultiDimensionLast()
//        {
//            float[] data = {1, 2, 3, 8, 3, 2};
//            int[] shape = {2, 3};
//            var tensor = new Syft.Tensor.FloatTensor(_ctrl, _data: data, _shape: shape);
//            var softmax = Syft.NN.Functional.Softmax(tensor);
//
//            float[] expected = {0.090f, 0.245f, 0.665f, 0.991f, 0.007f, 0.002f};
//            for (var i = 0; i < expected.Length; ++i)
//            {
//                Assert.AreEqual(expected[i], softmax[i], 1e-3);
//            }
//        }
//        
//        [Test]
//        public void SoftmaxMultiDimension0Th()
//        {
//            float[] data = {1, 2, 3, 8, 3, 2};
//            int[] shape = {2, 3};
//            var tensor = new Syft.Tensor.FloatTensor(_ctrl, _data: data, _shape: shape);
//            var softmax = Syft.NN.Functional.Softmax(tensor, 0);
//
//            // corresponds to
//            // 
//            // [ 
//            //   0.001, 0.269, 0.731
//            //   0.999, 0.731, 0.269
//            // ]
//            //
//            // Note that each column adds up to 1!
//            
//            float[] expected = {0.001f, 0.269f, 0.731f, 0.999f, 0.731f, 0.269f};
//            for (var i = 0; i < expected.Length; ++i)
//            {
//                Assert.AreEqual(expected[i], softmax[i], 1e-3);
//            }
//        }    
    }
}