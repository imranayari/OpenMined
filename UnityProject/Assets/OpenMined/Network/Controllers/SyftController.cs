using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using OpenMined.Syft.Tensor;
using OpenMined.Network.Utils;
using OpenMined.Syft.Layer;
using OpenMined.Syft.Model;
using Random = UnityEngine.Random;


namespace OpenMined.Network.Controllers
{
	public class SyftController
	{
		[SerializeField] private ComputeShader shader;

		private Dictionary<int, FloatTensor> tensors;
		private Dictionary<int, Model> models;
		private bool allow_new_tensors = true;

		public SyftController (ComputeShader _shader)
		{
			shader = _shader;

			tensors = new Dictionary<int, FloatTensor> ();
			models = new Dictionary<int, Model> ();
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

		public Model getModel(int index)
		{
			return models[index];
		}

		public ComputeShader GetShader ()
		{
			return shader;
		}

		public void RemoveTensor (int index)
		{
			Debug.LogFormat("<color=purple>Removing Tensor {0}</color>", index);
			var tensor = tensors [index];
			tensors.Remove (index);
			tensor.Dispose ();
		}

		public int addTensor (FloatTensor tensor)
		{
			if (allow_new_tensors)
			{
				//Debug.LogFormat("<color=green>Adding Tensor {0}</color>", tensor.Id);
				tensor.Controller = this;
				tensors.Add(tensor.Id, tensor);
				return (tensor.Id);
			}
			else
			{
				throw new Exception("Tried to allocate tensor");
			}
		}
		
		public int addModel (Model model)
		{
			models.Add (model.Id, model);
			return (model.Id);
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
			try
			{

				switch (msgObj.objectType)
				{
					case "tensor":
					{
						if (msgObj.objectIndex == 0 && msgObj.functionCall == "create")
						{
							FloatTensor tensor = new FloatTensor(this, _shape: msgObj.shape, _data: msgObj.data, _shader: this.Shader);
							Debug.LogFormat("<color=magenta>createTensor:{1}</color> {0}", string.Join(", ", tensor.Data), tensor.Id);
							return tensor.Id.ToString();
						}
						else if (msgObj.objectIndex > tensors.Count)
						{
							return "Invalid objectIndex: " + msgObj.objectIndex;
						}
						else
						{
							FloatTensor tensor = this.getTensor(msgObj.objectIndex);
							// Process message's function
							return tensor.ProcessMessage(msgObj, this);
						}
					}
					case "model":
					{
						if (msgObj.functionCall == "create")
						{
							string model_type = msgObj.tensorIndexParams[0];

							if (model_type == "linear")
							{
								Debug.LogFormat("<color=magenta>createModel:</color> {0} : {1} {2}", model_type,
									msgObj.tensorIndexParams[1], msgObj.tensorIndexParams[2]);
								Linear model = new Linear(this, int.Parse(msgObj.tensorIndexParams[1]), int.Parse(msgObj.tensorIndexParams[2]));
								return model.Id.ToString();
							}
							else if (model_type == "sigmoid")
							{
								Debug.LogFormat("<color=magenta>createModel:</color> {0}", model_type);
								Sigmoid model = new Sigmoid(this);
								return model.Id.ToString();
							}
							else if (model_type == "sequential")
							{
								Debug.LogFormat("<color=magenta>createModel:</color> {0}", model_type);
								Sequential model = new Sequential(this);
								return model.Id.ToString();
							}
                            else if (model_type == "tanh")
                            {
                                Debug.LogFormat("<color=magenta>createModel:</color> {0}", model_type);
                                Tanh model = new Tanh(this);
                                return model.Id.ToString();
                            }
						}
						else
						{
							Model model = this.getModel(msgObj.objectIndex);
							return model.ProcessMessage(msgObj, this);
						}
                        return "Unity Error: SyftController.processMessage: Command not found:" + msgObj.objectType + ":" + msgObj.functionCall;
					}
					case "controller":
					{
						if (msgObj.functionCall == "num_tensors")
						{
							return tensors.Count + "";
						} else if (msgObj.functionCall == "num_models")
						{
							return models.Count + "";
						} else if (msgObj.functionCall == "new_tensors_allowed")
						{
							
							
								Debug.LogFormat("New Tensors Allowed:{0}", msgObj.tensorIndexParams[0]);	
								if (msgObj.tensorIndexParams[0] == "True")
								{
									allow_new_tensors = true;
								} else if (msgObj.tensorIndexParams[0] == "False")
								{
									allow_new_tensors = false;
								}
								else
								{
									throw new Exception("Invalid parameter for new_tensors_allowed. Did you mean true or false?");
								}
							
							return allow_new_tensors + "";
						}
						return "Unity Error: SyftController.processMessage: Command not found:" + msgObj.objectType + ":" + msgObj.functionCall;
					}
						
				default:
						break;
				}
			}
			catch (Exception e)
			{
				Debug.LogFormat("<color=red>{0}</color>",e.ToString());
				return "Unity Error: " + e.ToString();
			}

			// If not executing createTensor or tensor function, return default error.
			return "Unity Error: SyftController.processMessage: Command not found:" + msgObj.objectType + ":" + msgObj.functionCall;            
		}
	}
}
