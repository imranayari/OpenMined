using System;
using System.Collections.Generic;
using System.Linq;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.NN
{
    public static class Functional
    {
        // TODO: Softmax will run on GPU, when below OPS have a GPU implementation!
        // TODO: Improve the implementation!!!
        public static FloatTensor Softmax(FloatTensor input, int dim = -1)
        {
            if (input.Shape.Length == 1)
            {
                var vCopy = input.Exp();
                return vCopy.Div(vCopy.Sum()[0], true);
            }
            
            //TODO: GPU support
            var gpu = false;
            if (input.DataOnGpu)
            {   
                input.Cpu();
                gpu = true;
            }
            
            //Below ops actually create a new copy!
            var y = (dim == -1) ? input.Shape.Length - 1 : dim;
            var lenY = input.Shape[y];
            var lenX = 1 * (input.Shape.Sum() - lenY);
            var viewShape = new int[] {lenX, lenY};
            var transposedShape = (int[]) input.Shape.Clone();
            
            var copy = input;
            if (dim != -1 && dim < (input.Shape.Length - 1))
            {
                // TODO: this is probably very inefficient
                for (var i = dim; i < input.Shape.Length - 1; i++)
                {
                    copy = copy.Transpose(i, i + 1);
                    transposedShape[i] = input.Shape[i + 1];
                    transposedShape[i + 1] = input.Shape[i];
                }
            }

            copy = (input.Shape.Length > 2) ? copy.View(viewShape).Exp() : copy.Exp();

            var expSums = copy.Sum(1);
            
            //TODO: fix the below line once we have matrix-vector division!
            var result = copy.emptyTensorCopy();
            for (var i = 0; i < lenX; i++)
            {
                for (var j = 0; j < lenY; j++)
                {
                    result[i, j] = copy[i, j] / expSums[i];
                }
            }
            
            if (dim != -1 && dim < (input.Shape.Length - 1))
            {
                //change the shape back to normal. this is probably very inefficient too.
                result = result.View(transposedShape);
                for (var i = input.Shape.Length - 1; i > dim ; i--)
                {
                    result = result.Transpose(i, i - 1);
                }
            }

            if (gpu)
            {
                result.Gpu(input.Shader);
            }

            return result;
        }
    }
}