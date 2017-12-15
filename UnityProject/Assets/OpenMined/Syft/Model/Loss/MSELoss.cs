using System.Runtime.InteropServices;
using OpenMined.Syft.Tensor;
using UnityEngine;

namespace OpenMined.Syft.Layer.Loss
{
    public class MSELoss
    {
        public static FloatTensor Value(FloatTensor input, FloatTensor target)
        {
            var diff = input.Sub(target);
            return diff.Pow(2);
            // TODO: sum
        }
    }
}