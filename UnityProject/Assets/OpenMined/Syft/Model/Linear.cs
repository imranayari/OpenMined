using JetBrains.Annotations;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Model
{
	public class Linear: Layer.Model
	{

		// private int _input;
		// private int _output;

		private readonly FloatTensor _weights;
		// private FloatTensor _bias;

		public Linear (SyftController controller, int input, int output)
		{
			// _input = input;
			// _output = output;

			int[] weightShape = { input, output };
			var weights = controller.RandomWeights(input * output);
			_weights = new FloatTensor(controller, _shape: weightShape, _data: weights, _autograd: true);

			// TODO: add bias when broadcast is available
			// int[] biasShape = {output};
			// _bias = new FloatTensor(controller, biasShape, _autograd: true);
		}

		protected override FloatTensor Forward(FloatTensor input)
		{
			return input.MM(_weights);
		}

		public override FloatTensor GetWeights()
		{
			return _weights;
		}
	}
}

