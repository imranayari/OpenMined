using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using OpenMined.Syft.Tensor;
using OpenMined.Network.Utils;


namespace OpenMined.Network.Controllers
{
	public class SyftController
	{
		[SerializeField] private ComputeShader shader;

		private Dictionary<int, FloatTensor> tensors;

		public SyftController (ComputeShader _shader)
		{
			shader = _shader;

			tensors = new Dictionary<int, FloatTensor> ();
		}

		public ComputeShader Shader {
			get { return shader; }
		}

		public float[] RandomWeights (int length)
		{
			Random.InitState (1);
			float[] syn0 = new float[length];
			for (int i = 0; i < length; i++) {
				syn0 [i] = 2 * Random.value - 1;
			}
			return syn0;
		}

		public FloatTensor getTensor (int index)
		{
			return tensors [index];
		}

		public ComputeShader GetShader ()
		{
			return shader;
		}

		public void RemoveTensor (int index)
		{
			var tensor = tensors [index];
			tensors.Remove (index);
			tensor.Dispose ();
		}

		public int addTensor (FloatTensor tensor)
		{
			tensor.Controller = this;
			tensors.Add (tensor.Id, tensor);
			return (tensor.Id);
		}

		public FloatTensor createZerosTensorLike(FloatTensor tensor) {
			FloatTensor new_tensor = tensor.Copy ();
			new_tensor.Zero_ ();
			return new_tensor;
		}

		public FloatTensor createOnesTensorLike(FloatTensor tensor) {
			FloatTensor new_tensor = tensor.Copy ();
			new_tensor.Zero_ ();
			new_tensor.Add ((float)1,true);
			return new_tensor;
		}

		public string processMessage (string json_message)
		{
			//Debug.LogFormat("<color=green>SyftController.processMessage {0}</color>", json_message);

			Command msgObj = JsonUtility.FromJson<Command> (json_message);

			switch (msgObj.objectType) {
			case "tensor":
				{
					if (msgObj.objectIndex == 0 && msgObj.functionCall == "create") {
						FloatTensor tensor = new FloatTensor (this,_shape:msgObj.shape, _data:msgObj.data, _shader:this.Shader);
						Debug.LogFormat ("<color=magenta>createTensor:</color> {0}", string.Join (", ", tensor.Data));
						return tensor.Id.ToString ();
					} else if (msgObj.objectIndex > tensors.Count) {
						return "Invalid objectIndex: " + msgObj.objectIndex;
					} else {
						FloatTensor tensor = this.getTensor (msgObj.objectIndex);
						// Process message's function
						return tensor.ProcessMessage (msgObj, this);
					}
				}
			default:
				break;                
			}
 
			// If not executing createTensor or tensor function, return default error.
			return "SyftController.processMessage: Command not found.";            
		}
	}
}
